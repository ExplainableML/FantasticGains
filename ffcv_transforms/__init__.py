from typing import List


from solo.ffcv_transforms.gaussian_blur import RandomGaussianBlur
from solo.ffcv_transforms.color_distortion import RandomColorDistortion
from solo.ffcv_transforms.color_jitter import RandomGrayscale
from solo.ffcv_transforms.solarization import RandomSolarization
from solo.ffcv_transforms.flip import RandomHorizontalFlip

def provide(augmentation_list: List[str]):
    augmentations = []
    for augmentation_str in augmentation_list:
        augmentation_params = {}
        if '_' in augmentation_str:
            augmentation_str, augmentation_params = augmentation_str.split('_')
            augmentation_params = eval(augmentation_params)
        augmentations.append(eval(augmentation_str)(**augmentation_params))
    return augmentations
