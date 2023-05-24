import json
import os

import hydra
import wandb
import torch
import logging

import numpy as np
import pandas as pd

from scipy.stats import entropy
from omegaconf import DictConfig, OmegaConf

from distillation.data import get_ffcv_val_loader
from distillation.models import init_timm_model
from distillation.dist_utils import get_batch_size, get_model, get_val_preds, parse_cfg
from distillation.flip_study import get_flips, get_flips_per_class, get_topk_class_sim


def get_predictions(student_name, teacher_name, cfg, student_pretrained=True, teacher_pretrained=True):
    """Get predictions of student and teacher on validation set

    :param student_name: name of student model
    :param teacher_name: name of teacher model
    :param cfg: config
    :param student_pretrained: whether to use pretrained student
    :param teacher_pretrained: whether to use pretrained teacher

    Returns: tuple (list of predictions, true labels)

    """
    # initialize the student (random) and the teacher (pretrained)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    student, cfg_s = init_timm_model(student_name, device, pretrained=student_pretrained)
    teacher, cfg_t = init_timm_model(teacher_name, device, pretrained=teacher_pretrained)
    # get batch size
    if cfg.optimizer.batch_size == 'auto':
        batch_size = get_batch_size(student_name, student_name, device, 'None')
        cfg.optimizer.batch_size = batch_size
    cfg = parse_cfg(cfg)
    # get val loader
    val_loader = get_ffcv_val_loader(cfg_s, device, cfg, batch_size=cfg.optimizer.batch_size)
    # get predictions
    preds_s, true_y = get_val_preds(student, val_loader, cfg_s, return_logits=False, return_truey=True)
    preds_t = get_val_preds(teacher, val_loader, cfg_t, return_logits=False, return_truey=False)
    return [preds_t, preds_s], true_y


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    tags = [cfg.mode]
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    # parse config
    OmegaConf.set_struct(cfg, False)

    models_list = pd.read_csv('files/contdist_model_list.csv')

    # draw random teacher and student from list of models
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
        'mode': cfg.mode
    }
    logging.info(f'Config: {config}')

    # init wandb
    wandb_id = wandb.util.generate_id()
    wandb.init(id=wandb_id, resume="allow", project=cfg.wandb.project, config=config, tags=tags)
    wandb.run.name = f'{teacher_name}>{student_name}'

    try:
        if cfg.mode == 'Random Baseline':  # random baseline (student model random init)
            preds, true_y = get_predictions(student_name, teacher_name, cfg, student_pretrained=False)
        else:  # both models are pretrained
            preds, true_y = get_predictions(student_name, teacher_name, cfg, student_pretrained=True)

        # calculate flips
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

        # get similarity of topk classes
        p_values = [2, 5, 20, 50]  # p values for topk classes
        k, avg_sim, max_sim, share = get_topk_class_sim(pos_flips, k=p_values)
        logging.info(f'Sim: {avg_sim}, Share: {share}')
        for i, p in enumerate(p_values):
            class_flips[f'top{p}%_avg_sim'] = avg_sim[i]
            class_flips[f'top{p}%_max_sim'] = max_sim[i]
            class_flips[f'top{p}%_share'] = share[i]
            class_flips[f'top{p}%_classes'] = k[i]

        # log results
        log = {**flips, **class_flips}
        logging.info(f'Logging results: {log}')
        wandb.log(log)

        # output sorted classes
        sorted_classes = np.argsort(pos_flips)[::-1]
        sorted_classes_dict = {sc: pos_flips[sc] for sc in sorted_classes}
        logging.info(f'Sorted Classes: {sorted_classes_dict}')

        # save results
        os.makedirs('prediction_flips', exist_ok=True)
        if cfg.mode == 'Random Baseline':
            filename = f'prediction_flips/{teacher_name}_{student_name}_rand.json'
        else:
            filename = f'prediction_flips/{teacher_name}_{student_name}.json'
        with open(filename, "w") as f:
            json.dump({'config': config, 'results': log}, f)

    except FileNotFoundError:
        logging.info(f'Logits for {teacher_name}>{student_name} not found')


if __name__ == '__main__':
    main()
