import os

import wandb
import torch
import logging

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

from distillation.dist_utils import AverageMeter, param_string, norm_batch


class BaseDisitllationTrainer:
    """This class implements the base distillation trainer

    Attributes:
        cfg: Experiment configuration
        teacher_name: The name of the teacher model
        student_name: The name of the student model
        student: The student model
        cfg_s: The configuration of the student model
        teacher: The teacher model
        cfg_t: The configuration of the teacher model
        module_list: The list of modules to be trained
        train_loader: The training data loader
        val_loader: The validation data loader
        device: The device to be used for training
        opt: The optimizer
        scheduler: The learning rate scheduler
        checkpoint_path: The path to the checkpoint
    """
    def __init__(self, cfg, teacher_name, student_name):
        self.cfg = cfg
        self.student_name = student_name
        self.teacher_name = teacher_name
        self.student, self.cfg_s = None, None
        self.teacher, self.cfg_t = None, None
        self.module_list = None

        self.train_loader, self.val_loader = None, None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.opt = None
        self.scheduler = None

        # generate the checkpoint path
        self.checkpoint_path = os.path.join(
            cfg.checkpoint.dir, cfg.data.dataset + f'_ffcv_{cfg.ffcv_dtype}', cfg.ffcv_augmentation, cfg.loss.name,
            param_string(cfg.loss), f'{teacher_name}>{student_name}', f'lr_{cfg.optimizer.lr}',
            f'batch_size_{cfg.optimizer.batch_size}', param_string(cfg.scheduler), f'seed_{cfg.seed}',param_string(cfg.on_flip))

    def get_imagenet_subset(self):
        """This method generates a stratified subset of the imagenet dataset

        Returns:

        """
        labels = []
        for _, label, idx in tqdm(self.train_loader):
            labels += label.cpu().tolist()

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1234)
        for _, subset_index in sss.split(np.zeros(len(labels)), labels):
            logging.info(f'Generated subset of size {len(subset_index)}')
            logging.info(Counter(np.array(labels)[subset_index]))
            subset_labels = np.array(labels)[subset_index]
            subset_indices = np.array(subset_index)

        for train_index, val_index in sss.split(subset_indices, subset_labels):
            train_indices = subset_indices[train_index]
            logging.info(f'Generated train-set of size {len(train_indices)}')
            logging.info(Counter(subset_labels[train_index]))
            val_indices = subset_indices[val_index]
            logging.info(f'Generated val-set of size {len(val_indices)}')
            logging.info(Counter(subset_labels[val_index]))

            np.save('create_ffcv_loaders/10p_subset_train.np', train_index)
            np.save('create_ffcv_loaders/10p_subset_val.np', val_index)

    def topk_div_test(self):
        """Assessment of the difference between the top-k and the total divergence

        Returns:

        """
        self.student.eval()
        self.teacher.eval()

        divergence = AverageMeter()
        topk_divergence = AverageMeter()
        agreement = AverageMeter()
        tk_weight_s = AverageMeter()
        tk_weight_t = AverageMeter()

        kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        for imgs, labels, idxs in tqdm(self.train_loader):
            # zero the parameter gradients
            self.opt.zero_grad()

            # pass batch images through the models
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    out_s = self.student(norm_batch(imgs, self.cfg_s))
                    out_s = nn.functional.log_softmax(out_s, dim=1)
                    out_t = self.teacher(norm_batch(imgs, self.cfg_t))
                    out_t = nn.functional.log_softmax(out_t, dim=1)
                    _, s_preds = torch.max(out_s, 1)
                    _, t_preds = torch.max(out_t, 1)

            agree = torch.div(torch.sum(torch.eq(t_preds, s_preds)), imgs.size(0))

            div = kl_loss(out_s, out_t)
            top_k, top_k_idx = torch.topk(out_t, self.cfg.k)
            topk_out_s = torch.randn((imgs.size(0), self.cfg.k), device=self.device)
            topk_out_t = torch.randn((imgs.size(0), self.cfg.k), device=self.device)
            for i, index in enumerate(top_k_idx):
                topk_out_s[i] = torch.index_select(out_s[i], 0, index)
                topk_out_t[i] = torch.index_select(out_t[i], 0, index)
            top10_div = kl_loss(topk_out_s, topk_out_t)

            t10w_s = torch.mean(torch.div(torch.sum(topk_out_s, dim=1), torch.sum(out_s, dim=1)))
            t10w_t = torch.mean(torch.div(torch.sum(topk_out_t, dim=1), torch.sum(out_t, dim=1)))

            divergence.update(div.item(), imgs.size(0))
            topk_divergence.update(top10_div.item(), imgs.size(0))
            agreement.update(agree.item(), imgs.size(0))
            tk_weight_s.update(t10w_s.item(), imgs.size(0))
            tk_weight_t.update(t10w_t.item(), imgs.size(0))

        stats = {'div': divergence.avg,
                 'topk_div': topk_divergence.avg,
                 'topk_div_rel': topk_divergence.avg / divergence.avg,
                 'agreement': agreement.avg * 100,
                 'topk_weight_s': tk_weight_s.avg * 100,
                 'topk_weight_t': tk_weight_t.avg * 100}
        logging.info(f'Stats: {stats}')
        wandb.log(stats)

    def check_accuracies(self, models_list, run, thresh=5):
        """Assert that inferred accuracies match the reported accuracies for both teacher and student model

        :param models_list: List of models from the timm library
        :param run: IDs of the teacher and student model
        :param thresh: Threshold for the difference between inferred and reported accuracy

        Returns:

        """
        # ensure that calculated accuracy match the reported accuracy for both models
        acc_diff_t = abs(self.teacher_acc[0] - models_list["modeltop1"][run[0]])
        acc_diff_s = abs(self.student_acc[0] - models_list["modeltop1"][run[1]])
        logging.debug(f'Teacher acc {self.teacher_acc[0]} (diff: {acc_diff_t})')
        logging.debug(f'Student acc {self.student_acc[0]} (diff: {acc_diff_s})')
        for (name, diff) in [[self.teacher_name, acc_diff_t], [self.student_name, acc_diff_s]]:
            assert diff < thresh, f'Calculated accuracy and reported accuracy for {name} by {round(diff, 2)} ' \
                                  f'\n sudent_cfg {self.cfg_s} \n teacher_cfg {self.cfg_t}'

    def load_from_checkpoint(self):
        """Load distillation run from checkpoint

        Returns: tuple (epoch, loss, wandb_id)
            - epoch: Latest epoch of the checkpoint
            - loss: Latest loss of the checkpoint
            - wandb_id: ID of the wandb run

        """
        checkpoint = torch.load(self.checkpoint_path + '/student_checkpoint.pt')
        if self.module_list is not None:
            for m, state_dict in enumerate(checkpoint['model_state_dict']):
                self.module_list[m].load_state_dict(state_dict)
        else:
            self.student.load_state_dict(checkpoint['model_state_dict'])
        self.theta_slow = checkpoint['theta_slow']
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        wandb_id = checkpoint['wandb_id']
        self.student_acc = checkpoint['student_acc_hist']
        self.teacher_acc = checkpoint['teacher_acc_hist']
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return epoch + 1, loss, wandb_id

    def save_to_checkpoint(self, epoch, loss, wandb_id):
        """Save distillation run to checkpoint

        :param epoch: Latest epoch
        :param loss: Latest loss
        :param wandb_id: ID of the wandb run

        Returns:

        """
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
                    'student_acc_hist': self.student_acc,
                    'teacher_acc_hist': self.teacher_acc},
                   self.checkpoint_path + '/student_checkpoint.pt')

