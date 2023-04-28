import torch
import timm
import logging

import numpy as np

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast
from scipy.stats import entropy

from .data import get_ffcv_val_loader


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
    from .dist_utils import norm_batch
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


def get_flips_per_class(preds_a, preds_b, true_y):
    pos_flips = np.zeros((len(true_y), 1000))
    neg_flips = np.zeros((len(true_y), 1000))
    for i in range(len(true_y)):
        pos_flips[i, true_y[i]] = preds_a[i] != true_y[i] and preds_b[i] == true_y[i]
        neg_flips[i, true_y[i]] = preds_a[i] == true_y[i] and preds_b[i] != preds_a[i]

    pos_flips = np.sum(pos_flips, axis=0)
    neg_flips = np.sum(neg_flips, axis=0)

    return pos_flips, neg_flips


def get_top_p_percent_classes(pos_class_flips, p):
    sorted_classes = np.argsort(pos_class_flips)[::-1]

    k = []
    for p_share in p:
        share, k_val = 0, 1
        while share < p_share:
            share = np.sum(pos_class_flips[sorted_classes[:k_val]]) / np.sum(pos_class_flips) * 100
            k_val += 1
        k.append(k_val)

    return k


def get_topk_class_sim(pos_class_flips, k=None, p=None):
    assert not(k is None and p is None), 'Please pass either k or p'
    sorted_classes = np.argsort(pos_class_flips)[::-1]

    if k is None:
        k = get_top_p_percent_classes(pos_class_flips, p)
    max_k = max(k)

    with open("files/imagenet1000_clsidx_to_labels.txt") as f:
        idx2label = eval(f.read())

    class_names = [idx2label[c] for c in sorted_classes[:max_k]]

    import clip
    device = torch.device('cuda')
    model, _ = clip.load('ViT-B/32', device, jit=False)
    text_tokens = clip.tokenize(class_names).to(device)
    with torch.no_grad():
        text_features = torch.nn.functional.normalize(model.encode_text(text_tokens), dim=-1)  # Top-k x Dim
    sims = text_features @ text_features.T
    tmp = sims.cpu().numpy()
    avg_sim = []
    max_sim = []
    share_of_flips = []
    for top_k in k:
        i, j = [], []
        for l in range(top_k):
            i += [l]*(top_k-l-1)
            j += range(l+1, top_k)
        avg_sim.append(sims[i, j].mean().item())
        max_sim.append(sims[i, j].max().item())
        share_of_flips.append(np.sum(pos_class_flips[sorted_classes[:top_k]]) / np.sum(pos_class_flips) * 100)
    return k, avg_sim, max_sim, share_of_flips


def get_dist_improvement(preds, zero_preds, teach_preds, true_y, p=[2, 5, 20, 50, 100]):
    pos_flips, _ = get_flips_per_class(zero_preds, teach_preds, true_y)
    dist_flips, _ = get_flips_per_class(zero_preds, preds, true_y)

    k = get_top_p_percent_classes(pos_flips, p)
    sorted_classes = np.argsort(pos_flips)[::-1]

    improvement = {}
    for i, top_k in enumerate(k):
        improvement[f'top{p[i]}%_improve'] = np.sum(dist_flips[sorted_classes[:top_k]])/np.sum(dist_flips)*100
    improvement['other_improve'] = 100-improvement['top100%_improve']
    improvement['ent_dist_flips'] = entropy(dist_flips)
    improvement['transfer_rate'] = np.sum(dist_flips[sorted_classes[:k[-1]]])/np.sum(pos_flips)*100

    return improvement