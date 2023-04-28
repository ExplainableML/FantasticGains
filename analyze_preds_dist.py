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

    model_name, model_type, model_params, model_acc = get_model(cfg.model_id, models_list, return_acc=True)

    config = {
        'counter': 1,
        'seed': cfg.seed,
        'model_name': model_name, 'model_type': model_type, 'model_params': model_params, 'model_acc': model_acc,
    }
    logging.info(f'Config: {config}')

    wandb_id = wandb.util.generate_id()
    wandb.init(id=wandb_id, resume="allow", project=cfg.wandb.project, config=config, tags=tags)
    wandb.run.name = f'{model_name}'

    try:
        logging.info('Loading logits')
        logits = np.load(f'model_logits/{model_name}_logits.npy')
        logging.info('Calculating predictions')
        preds = np.array(np.argmax(logits, axis=1))

        class_preds_counter = list(Counter(preds).values())
        logging.info(f'preds per class: {class_preds_counter}')
        # entropy
        ent_preds = entropy(class_preds_counter)

        log = {
            'ent_preds': ent_preds,
            'max_class_preds': max(class_preds_counter),
            'mean_class_preds': np.mean(class_preds_counter),
            'std_class_preds': np.std(class_preds_counter),
            'min_class_preds': min(class_preds_counter),
            'q25_class_preds': np.quantile(class_preds_counter, 0.25),
            'q75_class_preds': np.quantile(class_preds_counter, 0.75),
            'preds': list(preds),
        }

        logging.info(f'Logging results: {log}')
        wandb.log(log)

    except FileNotFoundError:
        logging.info(f'Logits for {model_name} not found')


if __name__ == '__main__':
    main()