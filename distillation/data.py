import os

import torch
import ffcv
import logging
import socket

import numpy as np

from ffcv.loader import OrderOption

import solo.ffcv_transforms as ffcv_transforms


def get_ffcv_val_loader(model_cfg, device, cfg, batch_size=None):
    """Initialize an ffcv dataloader for the validation dataset

    Args:
        model_cfg: Default config of the pretrained model
        device: Compute device
        cfg: Config

    Returns: data_loader
    """
    # initialize dataloaders
    decoder = ffcv.fields.rgb_image.CenterCropRGBImageDecoder(
        (model_cfg['input_size'][1], model_cfg['input_size'][2]), ratio=model_cfg['crop_pct'])
    order = OrderOption.SEQUENTIAL

    ffcv_dtype = torch.float16 if cfg.ffcv_dtype == 'float16' else torch.float32
    image_pipeline, label_pipeline, index_pipeline = [decoder], [], []
    image_pipeline.extend([
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(device, non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        ffcv.transforms.Convert(ffcv_dtype)
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

    loader = ffcv.loader.Loader(
        cfg.data.val_path,
        batch_size=batch_size if batch_size is not None else cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        order=order,
        os_cache=cfg.precache,
        drop_last=False,
        pipelines=pipeline,
        distributed=cfg.strategy == "ddp",
        seed=0
    )
    return loader


def get_ffcv_train_loader(model_cfg, device, cfg, batch_size=None):

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

    decoder = ffcv.fields.rgb_image.CenterCropRGBImageDecoder(
        (model_cfg['input_size'][1], model_cfg['input_size'][2]), ratio=model_cfg['crop_pct'])
    if cfg.strategy == 'ddp':
        order = OrderOption.RANDOM
    else:
        order = OrderOption.QUASI_RANDOM
    train_loader = []

    # os.environ['MASTER_ADDR'] = address
    # os.environ['MASTER_PORT'] = port

    # torch.distributed.init_process_group(
    #     "nccl", rank=self.base_gpu, world_size=world_size)
    # torch.cuda.set_device(self.base_gpu)
    ffcv_dtype = torch.float16 if cfg.ffcv_dtype == 'float16' else torch.float32
    for _ in range(num_loaders):
        image_pipeline, label_pipeline, index_pipeline = [decoder], [], []
        # For VicReg reduce strength!

        image_pipeline.extend(ffcv_transforms.provide(ffcv_augmentations['rest']))
        image_pipeline.extend([
            ffcv.transforms.ToTensor(),
            ffcv.transforms.ToDevice(device, non_blocking=True),
            ffcv.transforms.ToTorchImage(),
            ffcv.transforms.Convert(ffcv_dtype)
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
        logging.info(f'train batch size: {cfg.optimizer.batch_size}')
        data_loader = ffcv.loader.Loader(
            cfg.data.train_path,
            batch_size=batch_size if batch_size is not None else cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
            order=order,
            os_cache=cfg.precache,
            drop_last=False,
            pipelines=pipeline,
            distributed=cfg.strategy == "ddp",
            seed=0
        )
        train_loader.append(data_loader)
    train_loader = {i: x for i, x in enumerate(train_loader)}
    return train_loader[0]


def get_cls_pos_neg(train_loader):
    labels = []
    indices = []
    for _, y, idxs in train_loader:
        labels += y.cpu().tolist()
        indices += idxs.cpu().tolist()

    num_classes = len(np.unique(labels))
    num_samples = len(indices)
    idx_map = {idx: i for i, idx in enumerate(indices)}

    cls_positive = [[] for i in range(num_classes)]
    for i in range(num_samples):
        cls_positive[labels[i]].append(i)

    cls_negative = [[] for i in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            if j == i:
                continue
            cls_negative[i].extend(cls_positive[j])

    cls_positive = [np.asarray(cls_positive[i], dtype=np.int32) for i in range(num_classes)]
    cls_negative = [np.asarray(cls_negative[i], dtype=np.int32) for i in range(num_classes)]
    return cls_positive, cls_negative, idx_map, num_samples


def get_contrast_idx(index, target, cls_negative, k):
    index = index.cpu().tolist()
    target = target.cpu().tolist()
    sample_idx = []
    for i in range(len(index)):
        pos_idx = index[i]
        neg_idx = np.random.choice(cls_negative[target[i]], k, replace=True)
        sample_idx.append(np.hstack((np.asarray([pos_idx]), neg_idx)))
    return np.array(sample_idx)
