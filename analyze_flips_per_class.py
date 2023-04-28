import json
import os.path
import sys

import timm
import hydra
import wandb
import time
import torch

import numpy as np
import pandas as pd
from scipy.stats import entropy
from collections import Counter

from omegaconf import DictConfig, OmegaConf
from wandb import AlertLevel
import logging

from solo.args.pretrain import parse_cfg
from pytorch_lightning import seed_everything
from distillation.data import get_ffcv_val_loader, get_ffcv_train_loader, get_cls_pos_neg, get_contrast_idx
from distillation.models import init_timm_model, get_feature_dims, WeightingParams
from distillation.dist_utils import get_val_acc, get_val_preds, get_val_metrics, get_metrics, get_ensemble_metrics
from distillation.dist_utils import get_batch_size, AverageMeter, get_flip_masks, norm_batch
from distillation.dist_utils import random_search, grid_search, get_model, get_teacher_student_id
from distillation.contrastive.CRDloss import CRDLoss, DistillKL, DistillXE, Embed
from distillation.contrastive.SimpleContrastloss import SimpleContrastLoss
from distillation.trainer import BaseDisitllationTrainer
from distillation.flip_study import get_flips, get_flips_per_class, get_topk_class_sim


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    tags = [cfg.tag]
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    # parse config
    OmegaConf.set_struct(cfg, False)

    models_list = pd.read_csv('files/contdist_model_list.csv')

    if cfg.tag == 'Random Baseline':
        np.random.seed(cfg.seed)
        ids = np.random.choice(models_list.index, 2, replace=False)
        teacher_name, teacher_type, teacher_params, teacher_acc = get_model(ids[0], models_list, return_acc=True)
        student_name, student_type, student_params, student_acc = get_model(ids[1], models_list, return_acc=True)
    else:
        np.random.seed(cfg.seed)
        ids = np.random.choice(models_list.index, 2, replace=False)
        teacher_name, teacher_type, teacher_params, teacher_acc = get_model(ids[0], models_list, return_acc=True)
        student_name, student_type, student_params, student_acc = get_model(ids[1], models_list, return_acc=True)

    config = {
        'counter': 1,
        'seed': cfg.seed,
        'teacher_name': teacher_name, 'teacher_type': teacher_type, 'teacher_params': teacher_params, 'teacher_acc': teacher_acc,
        'student_name': student_name, 'student_type': student_type, 'student_params': student_params, 'student_acc': student_acc,
        'params_diff': teacher_params - student_params,
        'ts_diff': teacher_acc - student_acc,
        'tag': cfg.tag
    }
    logging.info(f'Config: {config}')

    wandb_id = wandb.util.generate_id()
    wandb.init(id=wandb_id, resume="allow", project=cfg.wandb.project, config=config, tags=tags)
    wandb.run.name = f'{teacher_name}>{student_name}'

    try:
        if cfg.tag == 'Random Baseline':
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            student, cfg_s = init_timm_model(student_name, device, pretrained=False)
            teacher, cfg_t = init_timm_model(teacher_name, device, pretrained=True)

            if cfg.optimizer.batch_size == 'auto':
                batch_size = get_batch_size(student_name, student_name, device, 'None')
                cfg.optimizer.batch_size = batch_size
            cfg = parse_cfg(cfg)

            val_loader = get_ffcv_val_loader(cfg_s, device, cfg, batch_size=cfg.optimizer.batch_size)

            preds_s, true_y = get_val_preds(student, val_loader, cfg_s, return_logits=False, return_truey=True)
            preds_t = get_val_preds(teacher, val_loader, cfg_t, return_logits=False, return_truey=False)
            preds = [preds_t, preds_s]
            logging.info(f'S preds: {Counter(preds_s)}')
            logging.info(f'T preds: {Counter(preds_t)}')
        else:
            logging.info('Loading logits')
            logits = [np.load(f'model_logits/{models_list["modelname"][i]}_logits.npy') for i in ids]
            logging.info('Loading gt-labels')
            true_y = np.load('model_logits/true_y.npy')
            logging.info('Calculating predictions')
            preds = np.array([np.argmax(l, axis=1) for l in logits])

        logging.info('Calculating Flips')
        pos_flips, neg_flips = get_flips_per_class(preds[1], preds[0], true_y)
        flips = get_flips(preds[1], preds[0], true_y)
        logging.info(f'total flips {flips}')

        # mean flips
        mean_pos_flips = np.mean(pos_flips)
        mean_neg_flips = np.mean(neg_flips)

        # std flips
        std_pos_flips = np.std(pos_flips)
        std_neg_flips = np.std(neg_flips)

        # entropy
        ent_pos_flips = entropy(pos_flips)
        ent_neg_flips = entropy(neg_flips)

        class_flips = {
            'pos_class_flips': list(pos_flips),
            'neg_class_flips': list(neg_flips),
            'mean_pos_class_flips': mean_pos_flips,
            'mean_neg_class_flips': mean_neg_flips,
            'std_pos_class_flips': std_pos_flips,
            'std_neg_class_flips': std_neg_flips,
            'ent_pos_class_flips': ent_pos_flips,
            'ent_neg_class_flips': ent_neg_flips
        }

        p_values = [2, 5, 20, 50]
        k, avg_sim, max_sim, share = get_topk_class_sim(pos_flips, k=p_values)
        logging.info(f'Sim: {avg_sim}, Share: {share}')
        for i, p in enumerate(p_values):
            class_flips[f'top{p}%_avg_sim'] = avg_sim[i]
            class_flips[f'top{p}%_max_sim'] = max_sim[i]
            class_flips[f'top{p}%_share'] = share[i]
            class_flips[f'top{p}%_classes'] = k[i]

        log = {**flips, **class_flips}
        logging.info(f'Logging results: {log}')
        wandb.log(log)

        sorted_classes = np.argsort(pos_flips)[::-1]
        sorted_classes_dict = {sc: pos_flips[sc] for sc in sorted_classes}
        logging.info(f'Sorted Classes: {sorted_classes_dict}')

        if cfg.tag == 'Random Baseline':
            filename = f'prediction_flips/{teacher_name}_{student_name}_rand.json'
        else:
            filename = f'prediction_flips/{teacher_name}_{student_name}.json'
        with open(filename, "w") as f:
            json.dump({'config': config, 'results': log}, f)

    except FileNotFoundError:
        logging.info(f'Logits for {teacher_name}>{student_name} not found')


if __name__ == '__main__':
    main()