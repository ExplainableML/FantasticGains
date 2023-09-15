from typing import List


from ffcv_transforms.gaussian_blur import RandomGaussianBlur
from ffcv_transforms.color_distortion import RandomColorDistortion
from ffcv_transforms.color_jitter import RandomGrayscale
from ffcv_transforms.solarization import RandomSolarization
from ffcv_transforms.flip import RandomHorizontalFlip

def provide(augmentation_list: List[str]):
    augmentations = []
    for augmentation_str in augmentation_list:
        augmentation_params = {}
        if '_' in augmentation_str:
            augmentation_str, augmentation_params = augmentation_str.split('_')
            augmentation_params = eval(augmentation_params)
        augmentations.append(eval(augmentation_str)(**augmentation_params))
    return augmentations
