import os
import torch
import ffcv
import logging
#import ffcv_transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.datasets import Caltech256, StanfordCars

import numpy as np

from ffcv.loader import OrderOption


def get_ffcv_val_loader(model_cfg, device, cfg, batch_size=None):
    """Initialize an ffcv dataloader for the validation dataset

    :param model_cfg: Default config of the pretrained model
    :param device: Compute device
    :param cfg: Config

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
    """Initialize an ffcv dataloader for the training dataset

    :param model_cfg: Default config of the pretrained model
    :param device: Compute device
    :param cfg: Config
    :param batch_size: Batch size

    :Returns: data_loader
    """
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
    """Get positive and negative samples for each class

    :param train_loader: Training data loader

    :Returns: cls_positive, cls_negative, idx_map, num_samples
    """
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
    """Get contrastive samples for each sample

    :param index: Sample index
    :param target: Sample target
    :param cls_negative: Negative samples for each class
    :param k: Number of negative samples

    :Returns: sample_idx
    """
    index = index.cpu().tolist()
    target = target.cpu().tolist()
    sample_idx = []
    for i in range(len(index)):
        pos_idx = index[i]
        neg_idx = np.random.choice(cls_negative[target[i]], k, replace=True)
        sample_idx.append(np.hstack((np.asarray([pos_idx]), neg_idx)))
    return np.array(sample_idx)


class CUBDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = os.path.join(root_dir, 'CUB_200_2011')
        self.transform = transform
        self.is_train = is_train
        self.image_paths, self.labels = self._load_image_paths_and_labels()

    def _load_image_paths_and_labels(self):
        image_paths = []
        labels = []
        train_set = []
        with open(os.path.join(self.root_dir, 'images.txt'), 'r') as f:
            for line in f:
                img_id, img_path = line.strip().split()
                image_paths.append(os.path.join(self.root_dir, 'images', img_path))
        with open(os.path.join(self.root_dir, 'image_class_labels.txt'), 'r') as f:
            for line in f:
                img_id, label = line.strip().split()
                labels.append(int(label) - 1)  # Labels in CUB dataset start from 1, we adjust them to start from 0.
        with open(os.path.join(self.root_dir, 'train_test_split.txt'), 'r') as f:
            for line in f:
                img_id, train = line.strip().split()
                train_set.append(int(train))
        assert len(image_paths) == len(labels) == len(train_set), 'Dataset length mismatch!'
        if self.is_train:
            image_paths = [image_paths[i] for i in range(len(image_paths)) if train_set[i] == 1]
            labels = [labels[i] for i in range(len(labels)) if train_set[i] == 1]
        else:
            image_paths = [image_paths[i] for i in range(len(image_paths)) if train_set[i] == 0]
            labels = [labels[i] for i in range(len(labels)) if train_set[i] == 0]
        logging.info(f'Loaded CUB. Number of images: {len(image_paths)}')
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


def get_cub_loader(cfg, model_cfg, is_train=True):
    """Get CUB dataset loader

    :param cfg: Config
    :param model_cfg: Model config
    :param is_train: Training or testing

    :Returns: data_loader
    """
    transform = []
    if is_train:
        transform.extend([
            transforms.RandomResizedCrop(model_cfg['input_size'][1]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform.extend([
            transforms.Resize(model_cfg['input_size'][1]),
            transforms.CenterCrop(model_cfg['input_size'][1]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    transform = transforms.Compose(transform)

    dataset = CUBDataset(cfg.data.data_path, transform=transform, is_train=is_train)
    data_loader = DataLoader(dataset, batch_size=cfg.optimizer.batch_size, shuffle=is_train, num_workers=cfg.data.num_workers)
    return data_loader


class CaltechDataset(Caltech256):
    def __init__(self, root_dir, transform=None, is_train=True):
        super(CaltechDataset, self).__init__(root_dir, transform=transform, download=True)
        self.is_train = is_train
        self.train_test_split()
        logging.info(f'Loaded CUB. Number of images: {len(self.y)}')
        logging.info(f'Number of classes: {len(np.unique(self.y))}')

    def train_test_split(self, test_size=1/6):
        # create a stratified split of the dataset
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.index, self.y,
                                                            stratify=self.y,
                                                            test_size=test_size, random_state=123)
        if self.is_train:
            self.index = X_train
            self.y = y_train
        else:
            self.index = X_test
            self.y = y_test

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if idx >= len(self):
            logging.error(f'Index {idx} out of range!')

        img = Image.open(
            os.path.join(
                self.root,
                "256_ObjectCategories",
                self.categories[self.y[idx]],
                f"{self.y[idx] + 1:03d}_{self.index[idx]:04d}.jpg",
            )
        ).convert("RGB")

        target = self.y[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, idx


def get_caltech_loader(cfg, model_cfg, is_train=True):
    """Get CUB dataset loader

    :param cfg: Config
    :param model_cfg: Model config
    :param is_train: Training or testing

    :Returns: data_loader
    """
    transform = []
    if is_train:
        transform.extend([
            transforms.RandomResizedCrop(model_cfg['input_size'][1]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform.extend([
            transforms.Resize(model_cfg['input_size'][1]),
            transforms.CenterCrop(model_cfg['input_size'][1]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    transform = transforms.Compose(transform)

    dataset = CaltechDataset(cfg.data.data_path, transform=transform, is_train=is_train)
    data_loader = DataLoader(dataset, batch_size=cfg.optimizer.batch_size, shuffle=is_train, num_workers=cfg.data.num_workers)
    return data_loader


class DomainNetInfographDataset(ImageFolder):
    def __init__(self, root, transform=None, is_train=True):
        root = os.path.join(root, 'domainset_infograph')
        super(DomainNetInfographDataset, self).__init__(root, transform=transform)
        self.root_dir = root
        if is_train:
            with open(os.path.join(root, 'infograph', 'infograph_train.txt'), 'r') as f:
                for line in f:
                    img_id, target = line.strip().split()
                    self.samples.append((os.path.join(root, img_id), int(target)))
        else:
            with open(os.path.join(root, 'infograph',  'infograph_test.txt'), 'r') as f:
                for line in f:
                    img_id, target = line.strip().split()
                    self.samples.append((os.path.join(root, img_id), int(target)))

        logging.info(f'Loaded DomainNet Infograph. Number of images: {len(self)}')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, index


def get_domainnet_loader(cfg, model_cfg, is_train=True):
    """Get CUB dataset loader

    :param cfg: Config
    :param model_cfg: Model config
    :param is_train: Training or testing

    :Returns: data_loader
    """
    transform = []
    if is_train:
        transform.extend([
            transforms.RandomResizedCrop(model_cfg['input_size'][1]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform.extend([
            transforms.Resize(model_cfg['input_size'][1]),
            transforms.CenterCrop(model_cfg['input_size'][1]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    transform = transforms.Compose(transform)

    dataset = DomainNetInfographDataset(cfg.data.data_path, transform=transform, is_train=is_train)
    data_loader = DataLoader(dataset, batch_size=cfg.optimizer.batch_size, shuffle=is_train, num_workers=cfg.data.num_workers)
    return data_loader


class CarsDataset(StanfordCars):
    def __init__(self, root_dir, transform=None, is_train=True):
        split = 'train' if is_train else 'test'
        super(CarsDataset, self).__init__(root_dir, split=split, transform=transform, download=True)
        logging.info(f'Loaded Stanford Cars. Number of images: {len(self.y)}')
        logging.info(f'Number of classes: {len(np.unique(self.y))}')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img, target = super(CarsDataset, self).__getitem__(idx)
        return img, target, idx


def get_cars_loader(cfg, model_cfg, is_train=True):
    """Get CUB dataset loader

    :param cfg: Config
    :param model_cfg: Model cfg
    :param is_train: Training or testing

    :Returns: data_loader
    """
    transform = []
    if is_train:
        transform.extend([
            transforms.RandomResizedCrop(model_cfg['input_size'][1]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform.extend([
            transforms.Resize(model_cfg['input_size'][1]),
            transforms.CenterCrop(model_cfg['input_size'][1]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    transform = transforms.Compose(transform)

    dataset = CarsDataset(cfg.data.data_path, transform=transform, is_train=is_train)
    data_loader = DataLoader(dataset, batch_size=cfg.optimizer.batch_size, shuffle=is_train, num_workers=cfg.data.num_workers)
    return data_loader
