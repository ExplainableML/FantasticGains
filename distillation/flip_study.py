import torch
import timm
import logging

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast

from .data import get_ffcv_val_loader
from .dist_utils import norm_batch


def get_predictions(model_a, cfg_a, data_loader, model_b, cfg_b, top_n=1):
    """Get predictions of up to two models

    Args:
        model_a:  Model a
        data_loader: Data loader
        model_b: Model b (optional)

    Returns: tuple
        - preds_a: Set of predictions for model_a
        - preds_b: Set of predictions for model_b (empty if only model a is provided)
        - true_y: Set of true labels
        - acc: List of calculated accuracies for model_a (and model_b)
    """
    true_y = []
    preds_a = []
    preds_b = []

    model_a.eval()
    model_b.eval()

    for imgs, y, idxs in data_loader:
        imgs = imgs.float()
        with torch.no_grad():
            with autocast():
                # pass batch images through the models
                outputs_a = model_a(norm_batch(imgs, cfg_a))
                outputs_b = model_b(norm_batch(imgs, cfg_b))
        # get class predictions from model outputs
        if top_n == 1:
            _, batch_preds_a = torch.max(outputs_a, 1)
            _, batch_preds_b = torch.max(outputs_b, 1)
            # record batch predictions
            preds_a += batch_preds_a.cpu().tolist()
            preds_b += batch_preds_b.cpu().tolist()
            # record true labels for the current batch
            true_y += y.cpu().tolist()
        else:
            _, batch_preds_a = torch.topk(outputs_a, top_n)
            _, batch_preds_b = torch.topk(outputs_b, top_n)
            for j in range(imgs.size(0)):
                preds_a.append(int(y[j] in batch_preds_a[j]))
                preds_b.append(int(y[j] in batch_preds_b[j]))
            true_y += [1]*imgs.size(0)

    # calculate accuracies based on the above generated predictions
    acc = [accuracy_score(true_y, preds_a)*100, accuracy_score(true_y, preds_b)*100]

    logging.debug(f'y: {y}')
    logging.debug(f'batch_p_a: {batch_preds_a}')
    logging.debug(f'preds_a: {preds_a[-imgs.size(0):]}')
    return preds_a, preds_b, true_y, acc


def get_flips(preds_a, preds_b, true):
    """calculate positive and negative flips for two sets of predictions

    Args:
        preds_a: Predictions of the first model
        preds_b: Predictions of the second model
        true: True labels

    Returns: dict
        - pos_abs: Absolute number of positive flips
        - neg_abs: Absolute number of negative flips
        - pos_rel: Relative number of positive flips (% of total predictions)
        - neg_rel: Relative number of negative flips (% of total predictions)
    """
    # check that all sets of labels have the same length
    assert len(preds_a) == len(preds_b) == len(true), "Flips cannot be computed: different number of instances"
    n_flips = 0
    p_flips = 0
    # iterate through instances and calculate positive/negative flips
    for i in range(len(preds_a)):
        # register a negative flip if prediction_a is correct and prediction_b is incorrect
        if preds_a[i] == true[i] and preds_b[i] != preds_a[i]:
            n_flips += 1
        # register a positive flip if prediction_a is incorrect and prediction_b is correct
        elif preds_a[i] != true[i] and preds_b[i] == true[i]:
            p_flips += 1
    # NOTE: neutral flips are not recorded
    return {'pos_abs': p_flips, 'neg_abs': n_flips, 'pos_rel': p_flips/len(true)*100, 'neg_rel': n_flips/len(true)*100}


def compare_models(modelname_a, modelname_b, cfg: DictConfig):
    """Compare predictions of two models and calculate positive and negative flips

    Args:
        modelname_a: Name of model_a
        modelname_b: Name of model_b
        cfg: Config

    Returns: tuple
        - flips: Dict of relative/absolut positive/negative flips
        - acc: List of model accuracies

    """
    # maintain a constant seed for inference
    seed_everything(123)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # initialize model_a (pretrained from timm)
    model_a = timm.create_model(modelname_a, pretrained=True)
    model_a.to(device)
    # get required image size
    cfg_a = model_a.default_cfg
    #if not cfg.performance.disable_channel_last:
    #    model_a = model_a.to(memory_format=torch.channels_last)

    # initialize model_a (pretrained from timm)
    model_b = timm.create_model(modelname_b, pretrained=True)
    model_b.to(device)
    # get required image size
    cfg_b = model_b.default_cfg
    #if not cfg.performance.disable_channel_last:
    #    model_b = model_b.to(memory_format=torch.channels_last)

    logging.info(f'Model default_cfgs: \n model_a: {model_a.default_cfg} \n model_b: {model_b.default_cfg}')

    # get dataloader for model_a
    loader_a = get_ffcv_val_loader(cfg_a, device, cfg)
    logging.info(f'res_a: {cfg_a["input_size"]}, res_b: {cfg_b["input_size"]}')

    preds_a, preds_b, true_y, acc = get_predictions(model_a, cfg_a, loader_a, model_b, cfg_b,  top_n=cfg.topn)


    # calculate flips
    flips = get_flips(preds_a, preds_b, true_y)

    # delete models and loader from gpu cache
    del model_a, model_b, loader_a
    torch.cuda.empty_cache()
    return flips, acc
