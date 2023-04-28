import os.path
import sys

import timm
import hydra
import wandb
import time
import torch

import numpy as np
import pandas as pd

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


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    # parse config
    OmegaConf.set_struct(cfg, False)


    models_list = pd.read_csv('files/contdist_model_list.csv')

    model_name, model_type, model_params = get_model(cfg.model_id, models_list)

    if os.path.exists(f'model_logits/{model_name}_logits.npy'):
        logging.info('Logits already saved')
        #sys.exit()

    model, cfg_m = init_timm_model(model_name, device)

    if cfg.optimizer.batch_size == 'auto':
        batch_size = get_batch_size(model_name, model_name, device, 'None')
        cfg.optimizer.batch_size = batch_size
    cfg = parse_cfg(cfg)

    val_loader = get_ffcv_val_loader(cfg_m, device, cfg, batch_size=cfg.optimizer.batch_size)

    logits, true_y = get_val_preds(model, val_loader, cfg_m, return_logits=True, return_truey=True)
    logging.info('Logits computed')
    np.save(f'model_logits/{model_name}_logits.npy', logits)
    np.save('model_logits/true_y.npy', true_y)
    logging.info('Logits saved')


if __name__ == '__main__':
    main()