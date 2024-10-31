"""
This script is licensed under the MIT License.
For more details, see the LICENSE file in the root directory of this repository.

(c) 2024 Lukas Thede
"""

import os
import sys

import hydra
import wandb
import time
import torch
import logging

import numpy as np
import pandas as pd

from omegaconf import DictConfig, OmegaConf
from wandb import AlertLevel

from main_distillation import DistillationTrainer
from distillation.dist_utils import get_val_acc, get_batch_size, param_string, get_val_metrics, contdist_grid_search, \
    get_val_preds, soup_student_weights, parse_cfg
from distillation.models import init_timm_model


class MultiDistillationTrainer(DistillationTrainer):
    """This class implements the multi teacher distillation trainer

        Attributes:
        cfg: Experiment configuration
        teacher_name: The name of the teacher model
        student_name: The name of the student model
        device: The device to be used for training
        checkpoint_path: The path to the checkpoint
        multi_teacher: whether to perform multi-teacher distillation
        module_list: list of modules to be trained (contrastive learning)
        criterion_list: list of loss functions (contrastive learning)
        opt: optimizer
        scheduler: learning rate scheduler (if specified)
        cfg_s: student model configuration
        cfg_t: teacher model configuration
        cfg_st: student-teacher model configuration (data partitioning)
        student: student model
        teacher: teacher model
        teachers: list of teacher models (if multi-teacher distillation)
        student_teacher: student-teacher model (data partitioning)
        scheduler: learning rate scheduler
        theta_slow: slow weights of the student model (weight interpolation)
        val_loader: validation data loader
        train_loader: training data loader
        cls_positive: positive samples (contrastive learning)
        cls_negative: negative samples (contrastive learning)
        idx_map: mapping from contrastive samples to their corresponding indices (contrastive learning)
        student_acc: student model accuracy
        teacher_acc: teacher model accuracy
        knowledge_gain: knowledge gain of the student model
        knowledge_loss: knowledge loss of the student model
        teacher_hist: teacher model history
        student_params: student model parameters
        teacher_params: teacher model parameters
        zero_preds_step: predictions of the student model before the current distillation step
    """
    def __init__(self, cfg, teacher_name, student_name):
        super().__init__(cfg, teacher_name, student_name)
        self.teacher_hist = []

        self.student_params = None
        self.teacher_params = None

        self.zero_preds_step = self.zero_preds

        self.checkpoint_path = os.path.join(
            cfg.checkpoint.dir, cfg.data.dataset + f'_ffcv_{cfg.ffcv_dtype}', cfg.ffcv_augmentation, cfg.loss.name,
            param_string(cfg.loss), f'{student_name}', f'curriculum_{cfg.multidist.curriculum}', f'sequential_{cfg.multidist.sequential}',  f'lr_{cfg.optimizer.lr}',
            f'batch_size_{cfg.optimizer.batch_size}', param_string(cfg.scheduler), f'seed_{cfg.seed}', param_string(cfg.on_flip))

    def update_teacher(self, name_new, params_new):
        """Update teacher model and student teacher model.

        :param name_new: name of new teacher model
        :param params_new: parameters of new teacher model

        :Returns:

        """
        # update teacher
        self.teacher_name = name_new
        self.teacher, self.cfg_t = init_timm_model(name_new, self.device)
        self.teacher_params = params_new
        self.teacher_acc.append(get_val_acc(self.teacher, self.val_loader, self.cfg_t))
        # update optimizer
        self.opt = torch.optim.SGD(self.student.parameters(), lr=self.cfg.optimizer.lr,
                                   momentum=self.cfg.optimizer.momentum,
                                   weight_decay=self.cfg.optimizer.weight_decay)
        # update student teacher
        self.student_teacher.load_state_dict(self.student.state_dict())
        self.zero_preds_step = get_val_preds(self.student, self.val_loader, self.cfg_s)

    def load_from_checkpoint(self):
        """Load from checkpoint.

        Returns: tuple (epoch, loss, wandb_id)
            - epoch: epoch to start from
            - loss: latest loss
            - wandb_id: wandb id of checkpoint

        """
        checkpoint = torch.load(self.checkpoint_path + '/student_checkpoint.pt')
        # load student weights
        if self.module_list is not None:
            for m, state_dict in enumerate(checkpoint['model_state_dict']):
                self.module_list[m].load_state_dict(state_dict)
        else:
            self.student.load_state_dict(checkpoint['model_state_dict'])
        # load student teacher weights
        if self.student_teacher is not None:
            self.student_teacher.load_state_dict(checkpoint['student_teacher_dict'])
        # load theta slow
        self.theta_slow = checkpoint['theta_slow']
        # load optimizer
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        # load epoch, loss, wandb-id
        epoch = checkpoint['epoch']
        if epoch == self.cfg.max_epochs - 1:
            epoch = 0
        else:
            epoch += 1
        self.teacher_hist = checkpoint['teacher_hist']
        loss = checkpoint['loss']
        wandb_id = checkpoint['wandb_id']
        self.student_acc = checkpoint['student_acc_hist']
        self.teacher_acc = checkpoint['teacher_acc_hist']
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return epoch, loss, wandb_id

    def save_to_checkpoint(self, epoch, loss, wandb_id):
        """Save to checkpoint.

        :param epoch: latest epoch
        :param loss: latest loss
        :param wandb_id: wandb id of checkpoint

        :Returns:

        """
        # save student weights
        if self.module_list is not None:
            model_state_dict = [self.module_list[m].state_dict() for m in
                                range(len(self.module_list) - 2)]
        else:
            model_state_dict = self.student.state_dict()
        # save scheduler
        scheduler_dict = self.scheduler.state_dict() if self.scheduler is not None else None
        # save student teacher weights
        student_teacher_dict = self.student_teacher.state_dict() if self.student_teacher is not None else None

        torch.save({'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'student_teacher_dict': student_teacher_dict,
                    'theta_slow': self.theta_slow,
                    'optimizer_state_dict': self.opt.state_dict(),
                    'scheduler_state_dict': scheduler_dict,
                    'loss': loss,
                    'wandb_id': wandb_id,
                    'teacher_hist': self.teacher_hist,
                    'student_acc_hist': self.student_acc,
                    'teacher_acc_hist': self.teacher_acc},
                   self.checkpoint_path + '/student_checkpoint.pt')

    def eval_mt_student(self, loss, t, e):
        """Evaluate student for multi-teacher distillation.

        :param loss: loss of current epoch
        :param t: current teacher step
        :param e: current epoch

        :Returns:

        """
        if self.cfg.multidist.sequential:
            s_metrics = get_val_metrics(self.student, self.teacher, self.val_loader, self.cfg_s, self.zero_preds,
                                        theta_slow=self.theta_slow, zero_preds_step=self.zero_preds_step)
            t_acc = get_val_acc(self.teacher, self.val_loader, self.cfg_t)
        else:
            s_metrics = get_val_metrics(self.student, self.teachers[0], self.val_loader, self.cfg_s, self.zero_preds,
                                        theta_slow=self.theta_slow, zero_preds_step=self.zero_preds_step)
            t_acc = get_val_acc(self.teachers[0], self.val_loader, self.cfg_t[0])
        self.student_acc.append(s_metrics['student_acc'])
        if e != 0 or t == 0:
            self.teacher_acc.append(t_acc)
        log = {**loss, **s_metrics}
        if e == self.cfg.max_epochs:
            log['dist_step_delta'] = s_metrics['dist_delta_step']
            log['dist_step_k_gain'] = s_metrics['knowledge_gain_step']
            log['dist_step_k_loss'] = s_metrics['knowledge_loss_step']
        log['lr'] = self.scheduler.get_lr()[0] if self.scheduler is not None else self.cfg.optimizer.lr
        logging.info(f'Log stats: {log}')
        wandb.log(log, step=t*self.cfg.max_epochs + e+1)

    def fit_xekl(self, current_step, wandb_id):
        """Distill student using cross entropy and KL divergence.

        :param current_step: current step of training
        :param wandb_id: wandb id of current run

        :Returns:

        """
        e_start, t = current_step
        for e in range(e_start, self.cfg.max_epochs):
            logging.info(f'Train step {e} for teacher {t}')
            loss = self.xe_kl_distill()
            logging.info('Save to checkpoint')
            self.save_to_checkpoint(e, loss, wandb_id)
            logging.info('Get student validation accuracy')
            self.eval_mt_student(loss, t, e)

    def soups_baseline(self):
        """Evaluate student using SOUPS baseline.

        :Returns:

        """
        # average (soup) the weights of the distilled students
        soup_weigts = soup_student_weights(self.student_name, self.teacher_name, self.cfg)
        # load soup weights into student
        self.student.load_state_dict(soup_weigts)
        # evaluate student
        self.eval_mt_student({}, 0, 0)


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    if cfg.search_id != 'None':
        cfg = contdist_grid_search(cfg, cfg.search_id)
    tags = [cfg.mode]
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    OmegaConf.set_struct(cfg, False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f'Student ID: {cfg.multidist.student_id}')

    # import list of models from the timm library
    models_list = pd.read_csv('files/timm_model_list.csv')

    # start run timer
    t_start = time.time()

    # get the student model
    student_name = models_list['modelname'][cfg.multidist.student_id]
    student_type = models_list['modeltype'][cfg.multidist.student_id]
    logging.info(f'Student Name: {student_name} ({student_type})')

    # get the teacher models
    if cfg.multidist.t_idxs is not None:
        t_idxs = cfg.multidist.t_idxs
    else:  # load random teacher curriculum from the list of models
        np.random.seed(cfg.multidist.t_seed)
        t_idxs = np.random.choice(models_list.index, cfg.multidist.n_teachers, replace=False)
    teacher_subset = models_list[t_idsx]
    if cfg.multidist.curriculum == 'asc':
        teacher_subset = teacher_subset.sort_values(by='modeltop1', ascending=True)
    elif cfg.multidist.curriculum == 'desc':
        teacher_subset = teacher_subset.sort_values(by='modeltop1', ascending=False)
    teachers = teacher_subset['modelname'].values
    teacher_types = teacher_subset['modeltype'].values
    teacher_params = teacher_subset['modelparams'].values
    logging.info(f'Teacher Names: {teachers}')

    # get suitable batch size for the student
    if cfg.optimizer.batch_size == 'auto':
        batch_size = get_batch_size(teachers[np.argmax(teacher_params)], student_name, device, cfg.loss.name)
    else:
        batch_size = cfg.optimizer.batch_size
    cfg.optimizer.batch_size = batch_size
    config['batch_size'] = batch_size

    # parse config
    cfg = parse_cfg(cfg)

    if cfg.multidist.sequential:  # initialize sequential trainer
        trainer = MultiDistillationTrainer(cfg, teachers[0], student_name)
    else:  # initialize parallel trainer
        trainer = MultiDistillationTrainer(cfg, teachers, student_name)

    config['student_name'] = student_name
    config['student_type'] = student_type
    config['teacher_list'] = teachers
    config['teacher_types'] = teacher_types
    logging.info(f'Run Config: {config}')

    trainer.student_params = models_list['modelparams'][cfg.multidist.student_id]
    trainer.teacher_params = models_list['modelparams'][t_idxs[0]]

    try:  # try to load from checkpoint
        if not cfg.checkpoint.enabled:
            raise FileNotFoundError
        e_start, loss, wandb_id = trainer.load_from_checkpoint()
        # initialize wandb logger
        wandb.init(id=wandb_id, resume="allow", project=cfg.wandb.project, config=config, tags=tags)
        wandb.run.name = f'{student_name}'
        logging.info(f'Loaded from checkpoint: {trainer.checkpoint_path}')
    except FileNotFoundError:
        # create checkpoint folder
        os.makedirs(trainer.checkpoint_path, exist_ok=True)
        # initialize wandb logger
        wandb_id = wandb.util.generate_id()
        wandb.init(id=wandb_id, project=cfg.wandb.project, config=config, tags=tags)
        wandb.run.name = f'{student_name}'
        e_start = 0
        wandb.log({'teacher_acc': trainer.teacher_acc[-1],
                   'student_acc': trainer.student_acc[-1],
                   'ts_diff': trainer.teacher_acc[-1]-trainer.student_acc[-1],
                   'student_params': models_list['modelparams'][cfg.multidist.student_id],
                   'teacher_params': models_list['modelparams'][t_idxs[0]]
                   }, step=0)

    if cfg.multidist.sequential:  # distill the teacher models sequentially
        for t, teacher in enumerate(teachers):
            logging.info(f'Distillation step {t}, teacher {teacher}')
            # check if teacher has already been distilled
            if teacher in trainer.teacher_hist:
                logging.info(f'Teacher {teacher} already distilled, continuing with next teacher')
                continue
            # update teacher model if not first teacher
            if t > 0:
                trainer.update_teacher(teacher, models_list['modelparams'][t_idxs[t]])
            # distill teacher
            if 'xekl' in cfg.loss.name:
                trainer.fit_xekl((e_start, t), wandb_id)
            elif cfg.loss.name in ['crd', 'cd']:
                trainer.fit_contrastive(e_start, wandb_id)
            else:
                raise Exception(f'Distillation approach {cfg.loss.name} not implemented')
            # add teacher to history
            trainer.teacher_hist.append(teacher)
            logging.info(f'Adding latest teacher to hist: {trainer.teacher_hist}')
            e_start = 0

            logging.info(f'Completed distillation for {teacher}>{student_name}')
            wandb.alert(
                title=f'COMPLETED: Distillation Iteration (seed {student_name})',
                text=f'Completed distillation for {teacher}>{student_name}'
                     f' ({round(time.time() - t_start, 2)}s)',
                level=AlertLevel.INFO
            )
    else:  # distill the teacher models in parallel
        t=0
        if 'Soup' in cfg.mode:  # soup baseline
            trainer.soups_baseline()
        elif 'xekl' in cfg.loss.name:
            trainer.fit_xekl((e_start, t), wandb_id)
        elif cfg.loss.name in ['crd', 'cd']:
            trainer.fit_contrastive(e_start, wandb_id)
        else:
            raise Exception(f'Distillation approach {cfg.loss.name} not implemented')
        # add teachers to history
        trainer.teacher_hist.append(teachers)

        logging.info(f'Completed distillation for {teachers}>{student_name}')
        wandb.alert(
            title=f'COMPLETED: Distillation ({student_name})',
            text=f'Completed distillation for {teachers}>{student_name}'
                 f' ({round(time.time() - t_start, 2)}s)',
            level=AlertLevel.INFO
        )


if __name__ == '__main__':
    main()
