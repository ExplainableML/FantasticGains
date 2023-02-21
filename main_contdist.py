import os
import sys

import hydra
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
import time
import torch
import logging

import numpy as np
import pandas as pd

from omegaconf import DictConfig, OmegaConf
from wandb import AlertLevel

from solo.args.pretrain import parse_cfg
from main_distillation import DistillationTrainer
from distillation.dist_utils import get_val_acc, get_batch_size, param_string
from distillation.models import init_timm_model


class ContinualDistillationTrainer(DistillationTrainer):
    def __init__(self, cfg, teacher_name, student_name):
        super().__init__(cfg, teacher_name, student_name)
        self.teacher_hist = []

        self.student_params = None
        self.teacher_params = None

        self.checkpoint_path = os.path.join(
            cfg.checkpoint.dir, cfg.data.dataset + f'_ffcv_{cfg.ffcv_dtype}', cfg.ffcv_augmentation, cfg.loss.name,
            param_string(cfg.loss), f'{student_name}', f'lr_{cfg.optimizer.lr}',
            f'batch_size_{cfg.optimizer.batch_size}', param_string(cfg.scheduler), f'seed_{cfg.seed}',param_string(cfg.on_flip))

    def update_teacher(self, name_new, params_new):
        self.teacher_name = name_new
        self.teacher, self.cfg_t = init_timm_model(name_new, self.device)
        self.teacher_params = params_new
        self.teacher_acc.append(get_val_acc(self.teacher, self.val_loader, self.cfg_t))
        self.opt = torch.optim.SGD(self.student.parameters(), lr=self.cfg.optimizer.lr, momentum=self.cfg.optimizer.momentum,
                                   weight_decay=self.cfg.optimizer.weight_decay)

    def load_from_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path + '/student_checkpoint.pt')
        if self.module_list is not None:
            for m, state_dict in enumerate(checkpoint['model_state_dict']):
                self.module_list[m].load_state_dict(state_dict)
        else:
            self.student.load_state_dict(checkpoint['model_state_dict'])
        self.theta_slow = checkpoint['theta_slow']
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
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
        if self.module_list is not None:
            model_state_dict = [self.module_list[m].state_dict() for m in
                                range(len(self.module_list) - 2)]
        else:
            model_state_dict = self.student.state_dict()

        scheduler_dict = self.scheduler.state_dict() if self.scheduler is not None else None

        torch.save({'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'theta_slow': self.theta_slow,
                    'optimizer_state_dict': self.opt.state_dict(),
                    'scheduler_state_dict': scheduler_dict,
                    'loss': loss,
                    'wandb_id': wandb_id,
                    'teacher_hist': self.teacher_hist,
                    'student_acc_hist': self.student_acc,
                    'teacher_acc_hist': self.teacher_acc},
                   self.checkpoint_path + '/student_checkpoint.pt')

    def check_accuracies(self, models_list, run, thresh=5):
        # ensure that calculated accuracy match the reported accuracy for both models
        acc_diff_t = abs(self.teacher_acc[0] - models_list["modeltop1"][run[0]])
        acc_diff_s = abs(self.student_acc[0] - models_list["modeltop1"][run[1]])
        logging.debug(f'Teacher acc {self.teacher_acc[0]} (diff: {acc_diff_t})')
        logging.debug(f'Student acc {self.student_acc[0]} (diff: {acc_diff_s})')
        for (name, diff) in [[self.teacher_name, acc_diff_t], [self.student_name, acc_diff_s]]:
            assert diff < thresh, f'Calculated accuracy and reported accuracy for {name} by {round(diff, 2)} ' \
                                  f'\n sudent_cfg {self.cfg_s} \n teacher_cfg {self.cfg_t}'

    def fit_xekl(self, current_step, wandb_id):
        e_start, t = current_step
        for e in range(e_start, self.cfg.max_epochs):
            logging.info(f'Train step {e} for teacher {t}')
            loss = self.xe_kl_distill()
            logging.info('Save to checkpoint')
            self.save_to_checkpoint(e, loss, wandb_id)
            logging.info('Get student validation accuracy')
            s_acc = get_val_acc(self.student, self.val_loader, self.cfg_s, theta_slow=self.theta_slow)
            t_acc = get_val_acc(self.teacher, self.val_loader, self.cfg_t)
            self.student_acc.append(s_acc)
            if e != 0 or t == 0:
                self.teacher_acc.append(t_acc)
            stats = {'dist_loss': loss[0],
                     'kl_loss': loss[1],
                     'xe_loss': loss[2],
                     'lr': self.scheduler.get_lr()[0] if self.scheduler is not None else self.cfg.optimizer.lr,
                     'student_acc': s_acc,
                     'teacher_acc': self.teacher_acc[-1],
                     'ts_diff': self.teacher_acc[-1] - self.student_acc[0],
                     'dist_delta': s_acc - self.student_acc[0],
                     'student_params': self.student_params,
                     'teacher_params': self.teacher_params}
            logging.info(f'Log stats: {stats}')
            wandb.log(stats, step=t*self.cfg.max_epochs + e+1)


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    tags = ['short-dist-long']
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # parse config
    OmegaConf.set_struct(cfg, False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f'Student ID: {cfg.contdist.student_id}')

    # import list of models from the timm library
    models_list = pd.read_csv('files/contdist_model_list.csv')

    # start run timer
    t_start = time.time()

    student_name = models_list['modelname'][cfg.contdist.student_id]
    student_type = models_list['modeltype'][cfg.contdist.student_id]
    logging.info(f'Studentname: {student_name} ({student_type})')

    t_idxs1 = [77, 66, 136, 157, 94, 43]
    np.random.seed(cfg.contdist.t_seed)
    t_idxs2 = np.random.choice(models_list.index, cfg.contdist.n_teachers, replace=False)
    t_idxs = np.concatenate((t_idxs1, t_idxs2))
    teachers = models_list.loc[t_idxs, 'modelname'].values
    teacher_types = models_list.loc[t_idxs, 'modeltype'].values
    logging.info(f'Teachernames: {teachers}')

    if cfg.optimizer.batch_size == 'auto':
        batch_size = get_batch_size(student_name, teachers[0], device)
    else:
        batch_size = cfg.optimizer.batch_size
    batch_size //= 2
    cfg.optimizer.batch_size = batch_size
    config['batch_size'] = batch_size
    cfg = parse_cfg(cfg)
    trainer = ContinualDistillationTrainer(cfg, teachers[0], student_name)
    trainer.check_accuracies(models_list, (t_idxs[0], cfg.contdist.student_id))

    config['student_name'] = student_name
    config['student_type'] = student_type
    config['teacher_list'] = teachers
    config['teacher_types'] = teacher_types
    logging.info(f'Run Config: {config}')

    trainer.student_params = models_list['modelparams'][cfg.contdist.student_id]
    trainer.teacher_params = models_list['modelparams'][t_idxs[0]]

    try:
        if not cfg.checkpoint.enabled:
            raise FileNotFoundError
        e_start, loss, wandb_id = trainer.load_from_checkpoint()
        # initialize wandb logger
        wandb.init(id=wandb_id, resume="allow", project=cfg.wandb.project, config=config, tags=tags)
        wandb.run.name = f'ContDist_{student_name}'
        logging.info(f'Loaded from checkpoint: {trainer.checkpoint_path}')
    except FileNotFoundError:
        # create checkpoint folder
        os.makedirs(trainer.checkpoint_path, exist_ok=True)
        # initialize wandb logger
        wandb_id = wandb.util.generate_id()
        wandb.init(id=wandb_id, project=cfg.wandb.project, config=config, tags=tags)
        wandb.run.name = f'ContDist_{student_name}'
        e_start = 0
        wandb.log({'teacher_acc': trainer.teacher_acc[-1],
                   'student_acc': trainer.student_acc[-1],
                   'ts_diff': trainer.teacher_acc[-1]-trainer.student_acc[-1],
                   'student_params': models_list['modelparams'][cfg.contdist.student_id],
                   'teacher_params': models_list['modelparams'][t_idxs[0]]
                   }, step=0)

    for t, teacher in enumerate(teachers):
        logging.info(f'Distillation step {t}, teacher {teacher}')
        if teacher in trainer.teacher_hist:
            logging.info(f'Teacher {teacher} already distilled, continuing with next teacher')
            continue
        if t > 0:
            trainer.update_teacher(teacher, models_list['modelparams'][t_idxs[t]])

        if cfg.loss.name in ['xekl', 'xekl_t', 'xekl_mcl']:
            trainer.fit_xekl((e_start, t), wandb_id)
        elif cfg.loss.name in ['crd', 'cd']:
            trainer.fit_contrastive(e_start, wandb_id)
        elif cfg.loss.name == 'test':
            trainer.kl_dist_test()
        else:
            raise Exception(f'Distillation approach {cfg.loss.name} not implemented')

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


if __name__ == '__main__':
    main()
