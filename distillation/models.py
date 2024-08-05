import torch
import timm
import logging

import torch.nn as nn


def freeze_all_but_linear(model):
    """Freeze all layers but the last linear layer

    :param model: model to freeze

    :Returns:
    """
    logging.info('Freeze non fc layers')
    for name, param in model.named_parameters():
        if 'fc' not in name and 'classifier' not in name and 'head' not in name:
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all layers

    :param model: model to unfreeze

    :Returns:
    """
    for name, param in model.named_parameters():
        param.requires_grad = True


def init_timm_model(name, device, split_linear=False, pretrained=True, return_cfg=True, num_classes=1000):
    """Initialize a timm model

    :param name: name of the model
    :param device: device to put the model on
    :param split_linear: whether to split the model into the feature extractor and linear layer
    :param pretrained: whether to load pretrained weights
    :param return_cfg: whether to return the model config
    :param num_classes: number of classes

    :Returns: model, model config (if return_cfg=True)
    """
    # initialize model
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    # get default model config
    cfg_m = model.default_cfg
    if split_linear:
        # initialize feature extractor only
        feature_extractor = timm.create_model(name, pretrained=True, num_classes=0)
        feature_extractor = nn.DataParallel(feature_extractor)
        feature_extractor.to(device, memory_format=torch.channels_last)
        feature_dims = get_feature_dims(feature_extractor, device)
        # initialize linear layer
        linear = nn.Sequential(nn.Linear(feature_dims, cfg_m['num_classes']))
        linear.load_state_dict(linear_state_dict(model))
        linear = nn.DataParallel(linear)
        linear.to(device, memory_format=torch.channels_last)
        del model
        if return_cfg:
            return feature_extractor, linear, cfg_m
        else:
            return feature_extractor, linear, cfg_m
    else:
        model = nn.DataParallel(model)
        if return_cfg:
            return model.to(device, memory_format=torch.channels_last), cfg_m
        else:
            return model.to(device, memory_format=torch.channels_last)


def get_feature_dims(model, device):
    """Get the feature dimensions of a model

    :param model: model to get the feature dimensions of
    :param device: device to put the model on

    :Returns: feature dimensions
    """
    input_shape = (3, 224, 224)
    inputs = torch.rand(*(2, *input_shape), device=device)
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            feat = model(inputs)
    logging.debug(f'feature dims: {feat.shape}')
    return feat.shape[1]


def linear_state_dict(model):
    """Get the state dict of the last linear layer of a model

    :param model: model to get the state dict of

    :Returns: state dict of the last linear layer
    """
    state_dict = model.state_dict()
    logging.debug(f'keys: {state_dict.keys()}')
    lin_keys = list(state_dict.keys())[-2:]
    logging.debug(f'lin_keys: {lin_keys}')
    lin_state_dict = {'0.weight': torch.squeeze(state_dict[lin_keys[0]]),
                      '0.bias': torch.squeeze(state_dict[lin_keys[1]])}
    return lin_state_dict


