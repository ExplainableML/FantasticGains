import torch
import logging

import numpy as np

from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast
from scipy.stats import entropy


def get_predictions(model_a, cfg_a, data_loader, model_b, cfg_b, top_n=1):
    """Get predictions of up to two models

    :param model_a:  Model a
    :param cfg_a: Config of model a
    :param data_loader: Data loader
    :param model_b: Model b (optional)
    :param cfg_b: Config of model b (optional)
    :param top_n: Number of top predictions to consider (default: 1)

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

    :param preds_a: Predictions of the first model
    :param preds_b: Predictions of the second model
    :param true: True labels

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


def get_flips_per_class(preds_a, preds_b, true_y):
    """Calculate positive and negative flips per class

    :param preds_a: Predictions of the first model
    :param preds_b: Predictions of the second model
    :param true_y: True labels

    Returns: tuple
        - pos_flips: Number of positive flips per class
        - neg_flips: Number of negative flips per class
    """
    pos_flips = np.zeros((len(true_y), 1000))
    neg_flips = np.zeros((len(true_y), 1000))
    for i in range(len(true_y)):
        pos_flips[i, true_y[i]] = preds_a[i] != true_y[i] and preds_b[i] == true_y[i]
        neg_flips[i, true_y[i]] = preds_a[i] == true_y[i] and preds_b[i] != preds_a[i]

    pos_flips = np.sum(pos_flips, axis=0)
    neg_flips = np.sum(neg_flips, axis=0)

    return pos_flips, neg_flips


def get_top_p_percent_classes(pos_class_flips, p):
    """Calculate the number of classes that account for the top p% of positive flips

    :param pos_class_flips: Number of positive flips per class
    :param p: Percentage of positive flips to be accounted for

    Returns: Set of classes that account for the top p% of positive flips
    """
    sorted_classes = np.argsort(pos_class_flips)[::-1]  # sort classes by number of positive flips

    k = []
    for p_share in p:
        share, k_val = 0, 1
        while share < p_share:
            share = np.sum(pos_class_flips[sorted_classes[:k_val]]) / np.sum(pos_class_flips) * 100
            k_val += 1
        k.append(k_val)

    return k


def get_topk_class_sim(pos_class_flips, k=None, p=None, save_path=None):
    """Calculate the similarity between a) the top-k classes containing the most porisitve flips (if k is passed) or
    b) the classes containing the top-p% of positive flips (if p is passed)

    :param pos_class_flips: Number of positive flips per class
    :param k: Number of classes to be considered
    :param p: Percentage of positive flips to be accounted for

    Returns: tuple
        - k: Set of classes used for the calculation
        - avg_sim: Average similarity between the classes
        - max_sim: Maximum similarity between the classes
        - share_of_flips: Share of positive flips accounted for by the classes
    """
    assert not(k is None and p is None), 'Please pass either k or p'
    sorted_classes = np.argsort(pos_class_flips)[::-1]  # sort classes by number of positive flips
    # get the classes that account for the top p% of positive flips (if p is passed)
    if k is None:
        k = get_top_p_percent_classes(pos_class_flips, p)
    max_k = max(k)
    # get the class names
    with open("files/imagenet1000_clsidx_to_labels.txt") as f:
        idx2label = eval(f.read())
    # get the class names of the top-k classes
    class_names = [idx2label[c] for c in sorted_classes[:max_k]]
    # calculate the similarity between the classes using CLIP embeddings of the class names
    import clip
    device = torch.device('cuda')
    model, _ = clip.load('ViT-B/32', device, jit=False)
    text_tokens = clip.tokenize(class_names).to(device)
    with torch.no_grad():
        text_features = torch.nn.functional.normalize(model.encode_text(text_tokens), dim=-1)  # Top-k x Dim
    sims = text_features @ text_features.T
    tmp = sims.cpu().numpy()
    if save_path is not None:
        np.save(f"{save_path}", tmp)
    avg_sim = []
    max_sim = []
    share_of_flips = []
    for c, top_k in enumerate(k):
        i, j = [], []
        for l in range(top_k):
            i += [l]*(top_k-l-1)
            j += range(l+1, top_k)
        tmp = sims[:top_k, :top_k].cpu().numpy()
        if save_path is not None:
            np.save(f"{save_path}_top{p[c]}p", tmp)
        avg_sim.append(sims[i, j].mean().item())
        max_sim.append(sims[i, j].max().item())
        share_of_flips.append(np.sum(pos_class_flips[sorted_classes[:top_k]]) / np.sum(pos_class_flips) * 100)
    return k, avg_sim, max_sim, share_of_flips


def get_dist_improvement(preds, zero_preds, teach_preds, true_y, p=[2, 5, 20, 50, 100]):
    """Calculate the improvement of the student relative to the share of positive flips

    :param preds: Predictions of the student model
    :param zero_preds: Predictions of the student before the distillation
    :param teach_preds: Predictions of the teacher model
    :param true_y: True labels
    :param p: Percentage of positive flips to be accounted for

    Returns: metrics of the improvement (dict)
    """
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
