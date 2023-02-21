import torch
import timm
import logging
import copy

import torch.nn as nn
from solo.utils.misc import make_contiguous


def freeze_all_but_linear(model):
    logging.info('Freeze non fc layers')
    for name, param in model.named_parameters():
        if 'fc' not in name and 'classifier' not in name:
            param.requires_grad = False


def unfreeze_all(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


def init_timm_model(name, device, split_linear=False):
    model = timm.create_model(name, pretrained=True)
    # get default model config
    cfg_m = model.default_cfg
    if split_linear:
        #feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor = timm.create_model(name, pretrained=True, num_classes=0)
        make_contiguous(feature_extractor)
        feature_extractor.to(device, memory_format=torch.channels_last)
        feature_dims = get_feature_dims(feature_extractor, device)

        linear = nn.Sequential(nn.Linear(feature_dims, cfg_m['num_classes']))
        linear.load_state_dict(linear_state_dict(model))
        make_contiguous(linear)
        linear.to(device, memory_format=torch.channels_last)
        del model
        return feature_extractor, linear, cfg_m
    else:
        make_contiguous(model)
        return model.to(device, memory_format=torch.channels_last), cfg_m


def get_feature_dims(model, device):
    input_shape = (3, 224, 224)
    inputs = torch.rand(*(2, *input_shape), device=device)
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            feat = model(inputs)
    logging.debug(f'feature dims: {feat.shape}')
    return feat.shape[1]


def linear_state_dict(model):
    state_dict = model.state_dict()
    logging.debug(f'keys: {state_dict.keys()}')
    lin_keys = list(state_dict.keys())[-2:]
    logging.debug(f'lin_keys: {lin_keys}')
    lin_state_dict = {'0.weight': torch.squeeze(state_dict[lin_keys[0]]),
                      '0.bias': torch.squeeze(state_dict[lin_keys[1]])}
    return lin_state_dict
