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
        
            image_pipeline, label_pipeline, index_pipeline = [decoder], [], []
            image_pipeline.extend(solo.ffcv_transforms.provide([
                'RandomHorizontalFlip_{"p":0.5}',
                'RandomColorDistortion_{"p":0.8,"strength":0.5}',
                'RandomGrayscale_{"p":0.2}',
                'RandomSolarization_{"p":0.2}'                
            ]))
            image_pipeline.extend([
                ffcv.transforms.ToTensor(),
                ffcv.transforms.ToDevice(device, non_blocking=True),
                ffcv.transforms.ToTorchImage(),
                ffcv.transforms.NormalizeImage(np.array(mean), np.array(std), np.float16)
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

            val_loader = ffcv.loader.Loader(
                '/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenette320/val_320_0.50_90.ffcv',
                batch_size=cfg.optimizer.batch_size,
                num_workers=cfg.data.num_workers,
                order=order,
                os_cache=False,
                drop_last=False,
                pipelines=pipeline,
                distributed=False,
                # distributed=cfg.strategy=='ddp',
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
        num_loaders = cfg.augmentations[0]['num_crops']
    
        decoder = ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((res, res))
        if cfg.strategy == 'ddp':
            order = OrderOption.RANDOM
        else:
            order = OrderOption.QUASI_RANDOM

        device = torch.device('cuda')
        
        train_loaders = []

        # os.environ['MASTER_ADDR'] = address
        # os.environ['MASTER_PORT'] = port

        # torch.distributed.init_process_group(
        #     "nccl", rank=self.base_gpu, world_size=world_size)
        # torch.cuda.set_device(self.base_gpu)
                
        for _ in range(num_loaders):
            image_pipeline, label_pipeline, index_pipeline = [decoder], [], []
            image_pipeline.extend(solo.ffcv_transforms.provide([
                'RandomHorizontalFlip_{"p":0.5}',
                'RandomColorDistortion_{"p":0.8,"strength":0.5}',
                'RandomGrayscale_{"p":0.2}',
                'RandomSolarization_{"p":0.2}'                
            ]))
            image_pipeline.extend([
                ffcv.transforms.ToTensor(),
                ffcv.transforms.ToDevice(device, non_blocking=True),
                ffcv.transforms.ToTorchImage(),
                ffcv.transforms.NormalizeImage(np.array(mean), np.array(std), np.float16)
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
                '/home/karsten_dl/Dropbox/Projects/Datasets/ffcv_imagenette320/train_320_0.50_90.ffcv',
                batch_size=cfg.optimizer.batch_size,
                num_workers=cfg.data.num_workers,
                order=order,
                os_cache=False,
                drop_last=False,
                pipelines=pipeline,
                distributed=False,
                # distributed=cfg.strategy=='ddp',
                seed=0
            )
            train_loaders.append(data_loader)
        train_loader = train_loaders[0]
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

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, cfg.method),
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
            logdir=os.path.join(cfg.checkpoint.dir, cfg.method),
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
