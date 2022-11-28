# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import inspect
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from solo.args.pretrain import parse_cfg
from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
)
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous

try:
    from solo.data.dali_dataloader import PretrainDALIDataModule, build_transform_pipeline_dali
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    seed_everything(cfg.seed)
    assert cfg.method in METHODS, f"Choose from {METHODS.keys()}"

    if cfg.data.num_large_crops != 2:
        assert cfg.method in ["wmse", "mae", "xent"]

    model = METHODS[cfg.method](cfg)
    make_contiguous(model)
    # can provide up to ~20% speed up
    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)

    # validation dataloader for when it is available
    if cfg.data.dataset == "custom" and (cfg.data.no_labels or cfg.data.val_path is None):
        val_loader = None
    elif cfg.data.dataset in ["imagenette320", "imagenet100", "imagenet"] and cfg.data.val_path is None:
        val_loader = None
    else:
        if cfg.data.format != 'ffcv':
            if cfg.data.format == "dali":
                val_data_format = "image_folder"
            else:
                val_data_format = cfg.data.format

            _, val_loader = prepare_data_classification(
                cfg.data.dataset,
                train_data_path=cfg.data.train_path,
                val_data_path=cfg.data.val_path,
                data_format=val_data_format,
                batch_size=cfg.optimizer.batch_size,
                num_workers=cfg.data.num_workers,
            )
        else:
            import solo.utils.constants
            import solo.ffcv_transforms
            
            import ffcv
            from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
            from ffcv.loader import OrderOption
            import numpy as np
            
            mean, std = solo.utils.constants.FFCV_MEANS_N_STD.get(cfg.data.dataset)
            res = cfg.augmentations[0]['crop_size']
            decoder = ffcv.fields.rgb_image.CenterCropRGBImageDecoder(
                (res, res), ratio=solo.utils.constants.DEFAULT_CROP_RATIO)
            order = OrderOption.SEQUENTIAL
            device = torch.device('cuda')
        
            ffcv_dtype = np.float16 if cfg.ffcv_dtype == 'float16' else np.float32
            image_pipeline, label_pipeline, index_pipeline = [decoder], [], []
            image_pipeline.extend([
                ffcv.transforms.ToTensor(),
                ffcv.transforms.ToDevice(device, non_blocking=True),
                ffcv.transforms.ToTorchImage(),
                ffcv.transforms.NormalizeImage(np.array(mean), np.array(std), ffcv_dtype)
            ])

            label_pipeline.extend([
                ffcv.fields.basics.IntDecoder(),
                ffcv.transforms.ToTensor(),
                ffcv.transforms.Squeeze(),
                ffcv.transforms.ToDevice(device, non_blocking=True)
            ])

            index_pipeline.extend([
                ffcv.fields.basics.IntDecoder(),
                ffcv.transforms.ToTensor(),
                ffcv.transforms.Squeeze(),
                ffcv.transforms.ToDevice(device, non_blocking=True)
            ])

            pipeline = {'image': image_pipeline, 'label': label_pipeline, 'index': index_pipeline}

            if cfg.strategy == "ddp":
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                torch.distributed.init_process_group(
                    "nccl", rank=0, world_size=len(cfg.devices))
                torch.cuda.set_device(0)
            
            val_loader = ffcv.loader.Loader(
                cfg.data.val_path,
                batch_size=cfg.optimizer.batch_size,
                num_workers=cfg.data.num_workers,
                order=order,
                os_cache=cfg.precache,
                drop_last=False,
                pipelines=pipeline,
                distributed=cfg.strategy=="ddp",
                seed=0
            )

    # pretrain dataloader
    if cfg.data.format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline_dali(
                        cfg.data.dataset, aug_cfg, dali_device=cfg.dali.device
                    ),
                    aug_cfg.num_crops,
                )
            )
        transform = FullTransformPipeline(pipelines)

        dali_datamodule = PretrainDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            transforms=transform,
            num_large_crops=cfg.data.num_large_crops,
            num_small_crops=cfg.data.num_small_crops,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
            dali_device=cfg.dali.device,
            encode_indexes_into_labels=cfg.dali.encode_indexes_into_labels,
        )
        dali_datamodule.val_dataloader = lambda: val_loader
    elif cfg.data.format == "ffcv":
        assert_str = f"The dataset {cfg.data.dataset} unfortunately is not ffcv-compatible."
        assert cfg.data.dataset in ['imagenette320', 'imagenet100', 'imagenet', 'cifar10', 'cifar100'], assert_str
        
        import solo.utils.constants
        import solo.ffcv_transforms
        
        import ffcv
        from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
        from ffcv.loader import OrderOption
        import numpy as np
        
        mean, std = solo.utils.constants.FFCV_MEANS_N_STD.get(cfg.data.dataset)

        res = cfg.augmentations[0]['crop_size']
        num_loaders = sum([aug_list['num_crops'] for aug_list in cfg.augmentations])
        
        aug_dicts = {
            'default': {
                'scale': (0.08, 1),
                'rest': [
                    'RandomHorizontalFlip_{"p":0.5}',
                    'RandomColorDistortion_{"p":0.8,"strength":1}',
                    'RandomGrayscale_{"p":0.2}'                    
                ]
            },
            'default_weak': {
                'scale': (0.08, 1),
                'rest': [
                    'RandomHorizontalFlip_{"p":0.5}',
                    'RandomColorDistortion_{"p":0.8,"strength":[0.5,0.5,0.25,0.5]}',
                    'RandomGrayscale_{"p":0.2}',
                    'RandomSolarization_{"p":0.1}'                     
                ]
            },
            'default_very_weak': {
                'scale': (0.3, 1),
                'rest': [
                    'RandomHorizontalFlip_{"p":0.5}',
                    'RandomColorDistortion_{"p":0.8,"strength":0.25}',
                    'RandomGrayscale_{"p":0.1}'                                         
                ]
            },
            'vicreg': {
                'scale': (0.2, 1),
                'rest': [
                    'RandomHorizontalFlip_{"p":0.5}',
                    'RandomColorDistortion_{"p":0.8,"strength":[0.5,0.5,0.25,0.5]}',
                    'RandomGrayscale_{"p":0.2}',
                    'RandomSolarization_{"p":0.1}'                                         
                ]
            },
            'minimal': {
                'scale': (0.08, 1),
                'rest': [
                    'RandomHorizontalFlip_{"p":0.5}'
                ]
            }                         
        }
        
        ffcv_augmentations = aug_dicts[cfg.ffcv_augmentation]
        scale = ffcv_augmentations['scale']
        
        decoder = ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((res, res), scale=scale)
        if cfg.strategy == 'ddp':
            order = OrderOption.RANDOM
        else:
            order = OrderOption.QUASI_RANDOM

        device = torch.device('cuda')
        
        train_loader = []

        # os.environ['MASTER_ADDR'] = address
        # os.environ['MASTER_PORT'] = port

        # torch.distributed.init_process_group(
        #     "nccl", rank=self.base_gpu, world_size=world_size)
        # torch.cuda.set_device(self.base_gpu)
        
        ffcv_dtype = np.float16 if cfg.ffcv_dtype == 'float16' else np.float32                        
        for _ in range(num_loaders):
            image_pipeline, label_pipeline, index_pipeline = [decoder], [], []
            # For VicReg reduce strength!
            
            image_pipeline.extend(solo.ffcv_transforms.provide(ffcv_augmentations['rest']))
            image_pipeline.extend([
                ffcv.transforms.ToTensor(),
                ffcv.transforms.ToDevice(device, non_blocking=True),
                ffcv.transforms.ToTorchImage(),
                ffcv.transforms.NormalizeImage(np.array(mean), np.array(std), ffcv_dtype)
            ])

            label_pipeline.extend([
                ffcv.fields.basics.IntDecoder(),
                ffcv.transforms.ToTensor(),
                ffcv.transforms.Squeeze(),
                ffcv.transforms.ToDevice(device, non_blocking=True)
            ])

            index_pipeline.extend([
                ffcv.fields.basics.IntDecoder(),
                ffcv.transforms.ToTensor(),
                ffcv.transforms.Squeeze(),
                ffcv.transforms.ToDevice(device, non_blocking=True)
            ])

            pipeline = {'image': image_pipeline, 'label': label_pipeline, 'index': index_pipeline}

            data_loader = ffcv.loader.Loader(
                cfg.data.train_path,
                batch_size=cfg.optimizer.batch_size,
                num_workers=cfg.data.num_workers,
                order=order,
                os_cache=cfg.precache,
                drop_last=True,
                pipelines=pipeline,
                distributed=cfg.strategy=="ddp",
                seed=0
            )
            train_loader.append(data_loader)
        train_loader = {i: x for i, x in enumerate(train_loader)}
    else:
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline(cfg.data.dataset, aug_cfg), aug_cfg.num_crops
                )
            )
        transform = FullTransformPipeline(pipelines)

        if cfg.debug_augmentations:
            print("Transforms:")
            print(transform)

        train_dataset = prepare_datasets(
            cfg.data.dataset,
            transform,
            train_data_path=cfg.data.train_path,
            data_format=cfg.data.format,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
        )
        train_loader = prepare_dataloader(
            train_dataset, batch_size=cfg.optimizer.batch_size, num_workers=cfg.data.num_workers
        )

        train_loader = {0: train_loader}
        
    # # Visual Debugging.
    # from IPython import embed; embed()
    # iterator = iter(zip(train_loader[0], train_loader[1]))
    # for k in range(3):
    #     out = next(iterator)
    #     rand_idcs = np.random.choice(cfg.optimizer.batch_size, 6, replace=False)
    #     samples_1 = out[0][0][rand_idcs].detach().cpu().numpy().astype(float)
    #     samples_2 = out[1][0][rand_idcs].detach().cpu().numpy().astype(float)
    #     samples_1 = np.clip((samples_1 * np.array(std).reshape(1, 3, 1, 1)) + np.array(mean).reshape(1, 3, 1, 1), 0, 255).astype(np.uint8)
    #     samples_2 = np.clip((samples_2 * np.array(std).reshape(1, 3, 1, 1)) + np.array(mean).reshape(1, 3, 1, 1), 0, 255).astype(np.uint8)
    #     import matplotlib.pyplot as plt
    #     f, axes = plt.subplots(2, len(rand_idcs))
    #     for i in range(len(rand_idcs)):
    #         axes[0, i].imshow(samples_1[i].transpose(1, 2, 0))
    #         axes[1, i].imshow(samples_2[i].transpose(1, 2, 0))
    #     f.set_size_inches(4 * len(rand_idcs), 4 * 2)
    #     f.tight_layout()
    #     f.savefig(f'sample_viz_{k}.png')
    #     plt.close()
    
    
    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    # from IPython import embed; embed()
    if cfg.wandb.group is None:
        if cfg.data.format == 'ffcv':
            checkpoint_path = os.path.join(
                cfg.checkpoint.dir, cfg.data.dataset + f'_ffcv_{cfg.ffcv_dtype}', cfg.ffcv_augmentation, cfg.method, cfg.name, f'seed_{cfg.seed}')
        else:
            checkpoint_path = os.path.join(
                cfg.checkpoint.dir, cfg.data.dataset, cfg.method, cfg.name, f'seed_{cfg.seed}')
    else:
        checkpoint_path = os.path.join(
            cfg.checkpoint_dir, cfg.data.dataset, cfg.wandb.group, cfg.name)    
        
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=checkpoint_path,
            max_hours=cfg.auto_resume.max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint

    callbacks = []

    if cfg.checkpoint.enabled:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            cfg,
            logdir=checkpoint_path,
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
        )
        callbacks.append(ckpt)

    if cfg.auto_umap.enabled:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            cfg.name,
            logdir=os.path.join(cfg.auto_umap.dir, cfg.method),
            frequency=cfg.auto_umap.frequency,
        )
        callbacks.append(auto_umap)

    # wandb logging
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        if cfg.wandb.watch_model:
            wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": DDPStrategy(find_unused_parameters=False)
            if cfg.strategy == "ddp"
            else cfg.strategy,
            'deterministic': cfg.deterministic
        }
    )
    trainer = Trainer(**trainer_kwargs)
    
    # print(OmegaConf.to_yaml(cfg))
    
    # fix for incompatibility with nvidia-dali and pytorch lightning
    # with dali 1.15 (this will be fixed on 1.16)
    # https://github.com/Lightning-AI/lightning/issues/12956
    try:
        from pytorch_lightning.loops import FitLoop

        class WorkaroundFitLoop(FitLoop):
            @property
            def prefetch_batches(self) -> int:
                return 1

        trainer.fit_loop = WorkaroundFitLoop(
            trainer.fit_loop.min_epochs, trainer.fit_loop.max_epochs
        )
    except:
        pass

    if cfg.data.format == "dali":
        trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
