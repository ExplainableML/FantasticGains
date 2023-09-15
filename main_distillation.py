import os
import sys

import hydra
#from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import wandb
import time
import torch
import logging

import pandas as pd
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf
from wandb import AlertLevel
import torch.backends.cudnn as cudnn

#from solo.args.pretrain import parse_cfg
from pytorch_lightning import seed_everything
from distillation.data import get_ffcv_val_loader, get_ffcv_train_loader, get_cls_pos_neg, get_contrast_idx, get_cub_loader, get_caltech_loader, get_cars_loader
from distillation.models import init_timm_model, get_feature_dims
from distillation.dist_utils import get_val_acc, get_val_preds, get_val_metrics, get_metrics
from distillation.dist_utils import get_batch_size, AverageMeter, get_flip_masks, norm_batch, parse_cfg, label_smoothing
from distillation.dist_utils import get_model, get_teacher_student_id, load_pretrain_weights
from distillation.contrastive.CRDloss import CRDLoss, DistillKL, DistillXE
from distillation.contrastive.SimpleContrastloss import SimpleContrastLoss
from distillation.trainer import BaseDisitllationTrainer
from distillation.dist_utils import CosineAnnealingLRWarmup


class DistillationTrainer(BaseDisitllationTrainer):
    """Trainer class to perform knowledge distillation from one teacher model to a student model

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
    """

    def __init__(self, cfg, teacher_name, student_name):
        super().__init__(cfg, teacher_name, student_name)
        seed_everything(cfg.seed)
        # check if multi-teacher distillation
        self.multi_teacher = not isinstance(teacher_name, str)

        if cfg.loss.name in ['crd', 'cd']:
            # initialize the teacher and student models
            student, student_lin, self.cfg_s = init_timm_model(student_name, self.device, split_linear=True, num_classes=cfg.data.num_classes)
            teacher, teacher_lin, self.cfg_t = init_timm_model(teacher_name, self.device, split_linear=True, num_classes=cfg.data.num_classes)
            s_dim = get_feature_dims(student, self.device)
            t_dim = get_feature_dims(teacher, self.device)

            self.module_list = nn.ModuleList([])
            self.criterion_list = nn.ModuleList([])
            trainable_list = nn.ModuleList([])

            self.module_list.append(student)  # add the student model to the module list
            self.module_list.append(student_lin)  # add the student linear layer to the module list
            trainable_list.append(student)  # add the student model to the trainable list
            trainable_list.append(student_lin)  # add the student linear layer to the trainable list

            if cfg.loss.name == 'crd':
                criterion_kd = CRDLoss(self.cfg, self.device, s_dim, t_dim)
                self.module_list.append(criterion_kd.embed_s)  # add the student embedding layers to the module list
                self.module_list.append(criterion_kd.embed_t)  # add the teacher embedding layers to the module list
                trainable_list.append(criterion_kd.embed_s)  # add the student embedding layers to the trainable list
                trainable_list.append(criterion_kd.embed_t)  # add the teacher embedding layers to the trainable list
            elif cfg.loss.name == 'cd':
                criterion_kd = SimpleContrastLoss()  # contrastive loss
            else:
                raise NotImplementedError

            criterion_cls = nn.CrossEntropyLoss()  # classification loss
            criterion_div = DistillKL(self.cfg.loss.kd_T)  # KL divergence loss, original knowledge distillation

            self.criterion_list.append(criterion_cls)  # classification loss
            self.criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
            if cfg.loss.name in ['crd', 'cd']:
                self.criterion_list.append(criterion_kd)  # other knowledge distillation loss

            self.opt = torch.optim.SGD(trainable_list.parameters(), lr=self.cfg.optimizer.lr,
                                       momentum=self.cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)

            self.module_list.append(teacher)  # add the teacher model to the module list
            self.module_list.append(teacher_lin)  # add the teacher linear layer to the module list

            if self.device == torch.device("cuda"):
                self.module_list.cuda()
                self.criterion_list.cuda()
                cudnn.benchmark = True

        else:
            # initialize the teacher and student models
            self.student, self.cfg_s = init_timm_model(student_name, self.device, num_classes=cfg.data.num_classes)
            if self.multi_teacher:  # if multi-teacher distillation
                self.teachers = nn.ModuleList([])
                self.cfg_t = []
                for teacher in teacher_name:
                    teacher, cfg_t = init_timm_model(teacher, self.device, num_classes=cfg.data.num_classes)
                    self.teachers.append(teacher)
                    self.cfg_t.append(cfg_t)
            else:  # if single teacher distillation
                self.teacher, self.cfg_t = init_timm_model(teacher_name, self.device, num_classes=cfg.data.num_classes)


            self.module_list = None
            self.criterion_list = None
            self.opt = torch.optim.SGD(self.student.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum,
                                       weight_decay=cfg.optimizer.weight_decay)

            # initialize the student-teacher model for multi-teacher distillation
            self.student_teacher, self.cfg_st = init_timm_model(student_name,
                                                                self.device, num_classes=cfg.data.num_classes) #if 'mt' in cfg.loss.name else (None, None)

            if 'imagenet' not in cfg.data.dataset:
                self.student = load_pretrain_weights(self.student, f'{os.path.join(cfg.checkpoint.dir, cfg.data.dataset, student_name + "_lin_ft" if cfg.freeze else student_name)}/{student_name}.pt')
                if self.student_teacher is not None:
                    self.student_teacher = load_pretrain_weights(self.student_teacher, f'{os.path.join(cfg.checkpoint.dir, cfg.data.dataset, student_name+ "_lin_ft" if cfg.freeze else student_name)}/{student_name}.pt')
                if self.multi_teacher:
                    for i, teacher in enumerate(self.teachers):
                        self.teachers[i] = load_pretrain_weights(teacher, f'{os.path.join(cfg.checkpoint.dir, cfg.data.dataset, teacher_name[i]+ "_lin_ft" if cfg.freeze else teacher_name[i])}/{teacher_name[i]}.pt')
                else:
                    self.teacher = load_pretrain_weights(self.teacher, f'{os.path.join(cfg.checkpoint.dir, cfg.data.dataset, teacher_name+ "_lin_ft" if cfg.freeze else teacher_name)}/{teacher_name}.pt')
            if cfg.teacher_pretrain == 'infograph':
                self.teacher = load_pretrain_weights(self.teacher, f'{os.path.join(cfg.checkpoint.dir, cfg.teacher_pretrain)}/{teacher_name}.pt', strict=False, drop_linear=True)
                logging.info(f'Loaded teacher model from {os.path.join(cfg.checkpoint.dir, cfg.teacher_pretrain)}/{teacher_name}.pt')
        # initialize the learning rate scheduler
        if cfg.scheduler.name == 'warmup_cosine':
            self.iter_per_epoch = len(self.train_loader)

            self.scheduler = CosineAnnealingLRWarmup(self.opt, cfg.max_epochs * self.iter_per_epoch,
                                                     warmup_iters=(cfg.scheduler.warmup / 100) * cfg.max_epochs * self.iter_per_epoch,
                                                     min_lr=cfg.scheduler.eta_min)
        else:
            self.scheduler = None

        # initialize theta slow for weight interpolation
        self.theta_slow = self.student.state_dict() if 'mcl' in cfg.loss.name else None

        # initialize the validation and train dataloaders
        if 'imagenet' in cfg.data.dataset:
            self.val_loader = get_ffcv_val_loader(self.cfg_s, self.device, cfg, batch_size=cfg.optimizer.batch_size)
            self.train_loader = get_ffcv_train_loader(self.cfg_s, self.device, cfg)
        elif cfg.data.dataset == 'CUB':
            self.val_loader = get_cub_loader(self.cfg, self.cfg_s, is_train=False)
            self.train_loader = get_cub_loader(self.cfg, self.cfg_s, is_train=True)
        elif cfg.data.dataset == 'caltech':
            self.val_loader = get_caltech_loader(self.cfg, self.cfg_s, is_train=False)
            self.train_loader = get_caltech_loader(self.cfg, self.cfg_s, is_train=True)
        elif cfg.data.dataset == 'cars':
            self.val_loader = get_cars_loader(self.cfg, self.cfg_s, is_train=False)
            self.train_loader = get_cars_loader(self.cfg, self.cfg_s, is_train=True)
        else:
            raise NotImplementedError

        if cfg.loss.name in ['crd', 'cd']:
            # get the class positive and negative indices for CRD
            self.cls_positive, self.cls_negative, self.idx_map, n_data = get_cls_pos_neg(self.train_loader)
            assert n_data == self.cfg.data.n_data, f'Actual dataset size ({n_data}) does not match config ({self.cfg.data.n_data})'
            # get the teacher and student validation accuracy
            self.student_acc = [
                get_val_acc(self.module_list[0], self.val_loader, self.cfg_s, linear=self.module_list[1])]
            self.teacher_acc = [
                get_val_acc(self.module_list[-2], self.val_loader, self.cfg_t, linear=self.module_list[-1])]
            self.zero_preds = get_val_preds(self.module_list[0], self.val_loader, self.cfg_s,
                                            linear=self.module_list[1])
        else:
            # get the student model validation predictions
            self.zero_preds = get_val_preds(self.student, self.val_loader, self.cfg_s, norm='imagenet' in cfg.data.dataset)
            # get the teacher and student validation accuracy
            self.student_acc = [get_val_acc(self.student, self.val_loader, self.cfg_s, norm='imagenet' in cfg.data.dataset)]
            if self.multi_teacher:
                self.teacher_acc = [[get_val_acc(teacher, self.val_loader, cfg_t, norm='imagenet' in cfg.data.dataset)] for teacher, cfg_t in
                                    zip(self.teachers, self.cfg_t)]
            else:
                self.teacher_acc = [get_val_acc(self.teacher, self.val_loader, self.cfg_t, norm='imagenet' in cfg.data.dataset)]
        self.knowledge_gain = [0]
        self.knowledge_loss = [0]

    def xe_kl_distill(self):
        """Perform one epoch of xe-kl knowledge distillation

        Returns: dict of losses and metrics

        """
        # set the student model to train mode
        self.student.train()
        # set the teacher model to eval mode
        if self.multi_teacher:
            for teacher in self.teachers:
                teacher.eval()
        else:
            self.teacher.eval()
        if self.student_teacher is not None:
            self.student_teacher.eval()

        # initialize the loss functions
        if 'xekl' in self.cfg.loss.name:
            kl_loss_fn = DistillKL(self.cfg.loss.kd_T)
        elif 'hinton' in self.cfg.loss.name:
            kl_loss_fn = DistillXE(self.cfg.loss.kd_T)
        else:
            raise NotImplementedError
        xe_loss_fn = torch.nn.CrossEntropyLoss()

        avg_loss = AverageMeter()
        avg_xe_loss = AverageMeter()
        avg_kl_loss = AverageMeter()
        teacher_input = AverageMeter()
        pos_flips = AverageMeter()
        neg_flips = AverageMeter()
        neut_flips = AverageMeter()
        train_k_gain = AverageMeter()
        train_k_loss = AverageMeter()
        train_dist_delta = AverageMeter()
        neg_to_teacher = AverageMeter()
        pos_to_teacher = AverageMeter()

        for j, (imgs, labels, idxs) in enumerate(self.train_loader):
            # zero the parameter gradients
            self.opt.zero_grad()
            labels = labels.to(self.device)

            # pass batch images through the models
            with torch.cuda.amp.autocast():
                # get the student model output
                out_s = self.student(norm_batch(imgs, self.cfg_s) if 'imagenet' in self.cfg.data.dataset else imgs)
                sp_s = nn.functional.softmax(out_s, dim=1)
                _, s_preds = torch.max(sp_s, 1)
                with torch.no_grad():
                    # get the teacher model output
                    if self.multi_teacher:
                        out_t, sp_t, t_preds = [], [], []
                        for t, teacher in enumerate(self.teachers):
                            o = teacher(norm_batch(imgs, self.cfg_t[t]) if 'imagenet' in self.cfg.data.dataset else imgs)
                            sp = nn.functional.softmax(o, dim=1)
                            _, pred = torch.max(sp, 1)
                            out_t.append(o)
                            sp_t.append(sp)
                            t_preds.append(pred)
                    else:
                        out_t = self.teacher(norm_batch(imgs, self.cfg_t) if 'imagenet' in self.cfg.data.dataset else imgs)
                        sp_t = nn.functional.softmax(out_t, dim=1)
                        _, t_preds = torch.max(sp_t, 1)
                    # get the student teacher model output
                    if self.student_teacher is not None:
                        out_st = self.student_teacher(norm_batch(imgs, self.cfg_st) if 'imagenet' in self.cfg.data.dataset else imgs)
                        sp_st = nn.functional.softmax(out_st, dim=1)
                        _, st_preds = torch.max(sp_st, 1)
                    else:
                        st_preds = None

            if self.multi_teacher:  # generate the teacher masks for each teacher in multi-teacher mode
                if 'dp' not in self.cfg.loss.name or 'most-conf' not in self.cfg.loss.strat or self.cfg.loss.k != 1000:
                    raise NotImplementedError

                t_mask = torch.zeros(imgs.size(0), device=self.device).to(torch.int)
                out_mt = torch.randn((imgs.size(0), self.cfg.loss.k), device=self.device)
                for i in range(imgs.size(0)):
                    outputs = torch.zeros(len(self.teachers) + 1, device=self.device)
                    if '_u' in self.cfg.loss.strat:
                        outputs[0] = torch.max(sp_st[i])
                        for t in range(len(self.teachers)):
                            outputs[t + 1] = torch.max(sp_t[t][i])
                        argmax = torch.argmax(outputs)
                        t_mask[i] = argmax - 1 if argmax > 0 else -1
                    else:
                        outputs[0] = sp_st[i][labels[i]]
                        for t in range(len(self.teachers)):
                            outputs[t + 1] = sp_t[t][i][labels[i]]
                        argmax = torch.argmax(outputs)
                        t_mask[i] = argmax - 1

                    out_mt[i] = out_t[t_mask[i]][i] if t_mask[i] >= 0 else out_st[i]
                kl_loss = kl_loss_fn(out_s.to(torch.float), out_mt)
                t_input = (torch.sum(t_mask >= 0) / imgs.size(0)) * 100
            else:  # generate the teacher mask for single teacher mode
                top_k, top_k_idx = torch.topk(out_t, self.cfg.loss.k)
                topk_out_s = torch.randn((imgs.size(0), self.cfg.loss.k), device=self.device)
                topk_out_t = torch.randn((imgs.size(0), self.cfg.loss.k), device=self.device)
                topk_out_st = torch.randn((imgs.size(0), self.cfg.loss.k), device=self.device)

                # get the top k outputs for each model
                for i, index in enumerate(top_k_idx):
                    topk_out_s[i] = torch.index_select(out_s[i], 0, index)
                    topk_out_t[i] = torch.index_select(out_t[i], 0, index)
                    if self.student_teacher is not None:
                        topk_out_st[i] = torch.index_select(out_st[i], 0, index)

                t_input = 0
                pos_mask, neg_mask, neut_mask = get_flip_masks(out_s, out_t, labels)  # get the prediction flip masks

                if 'dp' in self.cfg.loss.name:  # data partitioning
                    t_mask = torch.zeros((imgs.size(0)), device=self.device).to(torch.bool)
                    topk_out_mt = torch.randn((imgs.size(0), self.cfg.loss.k), device=self.device)
                    # get the teacher mask using the most confident strategy
                    if 'most-conf' in self.cfg.loss.strat:
                        for i in range(imgs.size(0)):
                            if '_u' in self.cfg.loss.strat:  # get the most confident teacher (unsupervised)
                                t_mask[i] = torch.max(sp_t[i]) > torch.max(sp_st[i])
                            else:  # get the most confident teacher (supervised)
                                t_mask[i] = sp_t[i][labels[i]] > sp_st[i][labels[i]]
                            topk_out_mt[i] = topk_out_t[i] if t_mask[i] else topk_out_st[i]
                        t_input = (torch.sum(t_mask) / imgs.size(0)) * 100  # get the percentage of teacher input

                    # get the teacher mask using the flip strategy
                    elif 'flip' in self.cfg.loss.strat:
                        if self.cfg.loss.strat == 'flip-avg':  # on neutral flips, average the teacher outputs
                            for i in range(imgs.size(0)):
                                if neut_mask[i]:
                                    topk_out_mt[i] = torch.mean(torch.stack((topk_out_t[i], topk_out_s[i]), dim=0), 0)
                                else:
                                    topk_out_mt[i] = topk_out_t[i] if pos_mask[i] else topk_out_st[i]
                        elif self.cfg.loss.strat == 'flip-st':  # on neutral flips, use the student teacher output
                            for i in range(imgs.size(0)):
                                topk_out_mt[i] = topk_out_t[i] if pos_mask[i] else topk_out_st[i]
                            t_input = (torch.sum(pos_mask).item() / imgs.size(0)) * 100
                        elif self.cfg.loss.strat == 'flip-t':  # on neutral flips, use the teacher output
                            for i in range(imgs.size(0)):
                                topk_out_mt[i] = topk_out_st[i] if neg_mask[i] else topk_out_t[i]
                            t_input = (1 - torch.sum(neg_mask).item() / imgs.size(0)) * 100
                        elif self.cfg.loss.strat == 'self-dist':  # self distillation baseline
                            topk_out_mt = topk_out_st
                    else:
                        raise NotImplementedError
                    kl_loss = kl_loss_fn(topk_out_s, topk_out_mt)  # get the kl loss

                else:
                    # get the kl loss for different flip strategies
                    if self.cfg.on_flip.pos == 'distill' and self.cfg.on_flip.neg == 'distill' and self.cfg.on_flip.neut == 'distill':
                        kl_loss = kl_loss_fn(topk_out_s, topk_out_t)
                    elif self.cfg.on_flip.pos == 'distill' and self.cfg.on_flip.neut == 'distill':
                        kl_loss = kl_loss_fn(topk_out_s[torch.logical_not(neg_mask)],
                                             topk_out_t[torch.logical_not(neg_mask)])
                    elif self.cfg.on_flip.pos == 'distill':
                        kl_loss = kl_loss_fn(topk_out_s[pos_mask], topk_out_t[pos_mask])
                    if torch.isnan(kl_loss):
                        kl_loss = torch.zeros(1, device=self.device)

            if self.cfg.loss.label_smoothing > 0:
                sp_s = label_smoothing(sp_s, self.cfg.loss.label_smoothing)  # apply label smoothing

            xe_loss = xe_loss_fn(sp_s, labels)  # get the cross entropy loss
            # get the final loss
            loss = float(self.cfg.loss.alpha) * kl_loss + (1 - float(self.cfg.loss.alpha)) * xe_loss

            loss.backward()
            self.opt.step()

            # update the slow weights for weight interpolation
            if self.theta_slow is not None and j % self.cfg.loss.N == 0:
                theta = self.student.state_dict()
                for key in theta.keys():
                    self.theta_slow[key] = self.cfg.loss.tau * self.theta_slow[key] + (1 - self.cfg.loss.tau) * theta[
                        key]

            # update the training metrics
            if self.multi_teacher:  # multi-teacher mode (calculate metrics based on the firt teacher)
                train_metrics = get_metrics(st_preds.cpu().tolist(), s_preds.cpu().tolist(), t_preds[0].cpu().tolist(),
                                            labels.cpu().tolist(), train=True)

                avg_loss.update(loss.item(), imgs.size(0))
                avg_kl_loss.update(kl_loss.item(), imgs.size(0))
                avg_xe_loss.update(xe_loss.item(), imgs.size(0))
                teacher_input.update(t_input, imgs.size(0))
                train_k_gain.update(train_metrics['knowledge_gain'], imgs.size(0))
                train_k_loss.update(train_metrics['knowledge_loss'], imgs.size(0))
                train_dist_delta.update(train_metrics['dist_delta'], imgs.size(0))

            else: # single-teacher mode
                train_metrics = get_metrics(st_preds.cpu().tolist(), s_preds.cpu().tolist(), t_preds.cpu().tolist(),
                                            labels.cpu().tolist(), train=True)

                avg_loss.update(loss.item(), imgs.size(0))
                avg_kl_loss.update(kl_loss.item(), imgs.size(0))
                avg_xe_loss.update(xe_loss.item(), imgs.size(0))
                teacher_input.update(t_input, imgs.size(0))
                pos_flips.update(torch.sum(pos_mask).item() / imgs.size(0), imgs.size(0))
                neg_flips.update(torch.sum(neg_mask).item() / imgs.size(0), imgs.size(0))
                neut_flips.update(torch.sum(neut_mask).item() / imgs.size(0), imgs.size(0))
                train_k_gain.update(train_metrics['knowledge_gain'], imgs.size(0))
                train_k_loss.update(train_metrics['knowledge_loss'], imgs.size(0))
                train_dist_delta.update(train_metrics['dist_delta'], imgs.size(0))
                if 'dp' in self.cfg.loss.name:  # if using the dp loss, update the dp metrics
                    neg_to_teacher.update(torch.sum(t_mask[neg_mask] / torch.sum(neg_mask)), imgs.size(0))
                    pos_to_teacher.update(torch.sum(t_mask[pos_mask] / torch.sum(pos_mask)), imgs.size(0))
                else:
                    neg_to_teacher.update(0, imgs.size(0))
                    pos_to_teacher.update(0, imgs.size(0))

        if self.scheduler is not None:
            self.scheduler.step()

        out = {'dist_loss': avg_loss.avg, 'kl_loss': avg_kl_loss.avg, 'xe_loss': avg_xe_loss.avg,
               'teacher_input': teacher_input.avg, 'pos_to_teacher': pos_to_teacher.avg,
               'neg_to_teacher': neg_to_teacher.avg,
               'k_gain_train': train_k_gain.avg, 'k_loss_train': train_k_loss.avg,
               'dist_delta_train': train_dist_delta.avg,
               'pos_flips': pos_flips.avg, 'neg_flips': neg_flips.avg, 'neut_flips': neut_flips.avg}

        return out

    def contrastive_distill(self):
        """Perform one epoch of contrastive distillation

        Returns: dict of losses and metrics
        """
        # set modules as train()
        for module in self.module_list:
            module.train()
        # set teacher as eval()
        self.module_list[-1].eval()

        # set losses
        criterion_cls = self.criterion_list[0]
        criterion_div = self.criterion_list[1]
        criterion_kd = self.criterion_list[2]

        # set student and teacher models
        model_s = self.module_list[0]
        model_s_lin = self.module_list[1]
        model_t = self.module_list[-2]
        model_t_lin = self.module_list[-1]

        losses = AverageMeter()
        losses_cls = AverageMeter()
        losses_div = AverageMeter()
        losses_kd = AverageMeter()

        for input, target, index in self.train_loader:
            if self.cfg.loss.name == 'crd':  # get contrastive indices
                index = torch.tensor([self.idx_map[idx] for idx in index.cpu().tolist()], device=self.device)
                contrast_idx = get_contrast_idx(index, target, self.cls_negative, self.cfg.loss.nce_k)
                contrast_idx = torch.tensor(contrast_idx, device=self.device).int()

            with torch.cuda.amp.autocast():
                # get student and teacher features
                feat_s = model_s(norm_batch(input, self.cfg_s))
                logit_s = model_s_lin(feat_s)
                with torch.no_grad():
                    feat_t = model_t(norm_batch(input, self.cfg_t))
                    logit_t = model_t_lin(feat_t)
                    feat_t = feat_t.detach()

            loss_cls = criterion_cls(logit_s, target)  # classification loss
            loss_div = criterion_div(logit_s, logit_t)  # KL-divergence loss

            f_s = feat_s.float()
            f_t = feat_t.float()
            # get contrastive loss
            if self.cfg.loss.name == 'crd':
                loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
            else:
                loss_kd = criterion_kd(f_s, f_t)

            # get total loss
            loss = self.cfg.loss.gamma * loss_cls + self.cfg.loss.alpha * loss_div + self.cfg.loss.beta * loss_kd

            losses.update(loss.item(), input.size(0))
            losses_cls.update(loss_cls.item(), input.size(0))
            losses_div.update(loss_div.item(), input.size(0))
            losses_kd.update(loss_kd.item(), input.size(0))

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        if self.scheduler is not None:
            self.scheduler.step()

        out = {'dist_loss': losses.avg, 'kl_loss': losses_div.avg, 'xe_loss': losses_cls.avg, 'cd_loss': losses_kd.avg}

        return out

    def eval_student(self, loss):
        """Evaluate the student model and log metrics

        :param loss: training loss

        Returns:

        """
        # get validation metrics for the student
        if self.cfg.loss.name in ['crd', 'cd'] or ('dp' in self.cfg.loss.name and 'adaptive' in self.cfg.loss.strat):
            s_metrics = get_val_metrics(self.module_list[0], self.module_list[-2], self.val_loader, self.cfg_s,
                                        self.zero_preds,
                                        linear_s=self.module_list[1], linear_t=self.module_list[-1], norm='imagenet' in self.cfg.data.dataset)
        else:
            if self.multi_teacher:
                s_metrics = get_val_metrics(self.student, self.teachers[0], self.val_loader, self.cfg_s,
                                            self.zero_preds,
                                            theta_slow=self.theta_slow, norm='imagenet' in self.cfg.data.dataset)
            else:
                s_metrics = get_val_metrics(self.student, self.teacher, self.val_loader, self.cfg_s, self.zero_preds,
                                            theta_slow=self.theta_slow, norm='imagenet' in self.cfg.data.dataset)
        logging.info(f'Metrics: {s_metrics}')
        t_acc = self.teacher_acc[-1]
        # update accuracies
        self.student_acc.append(s_metrics['student_acc'])
        self.teacher_acc.append(t_acc)
        # log metrics
        log = {**loss, **s_metrics}
        log['lr'] = self.scheduler.get_lr()[0] if self.scheduler is not None else self.cfg.optimizer.lr
        logging.info(f'Log stats: {log}')
        wandb.log(log)

    def fit_xekl(self, e_start, wandb_id):
        """Perform xe-kl distillation

        :param e_start: Start epoch
        :param wandb_id: ID of the wandb run

        Returns:

        """
        for e in range(e_start, self.cfg.max_epochs):
            logging.info(f'Train step {e}')
            loss = self.xe_kl_distill()  # perform distillation
            logging.info('Save to checkpoint')
            self.save_to_checkpoint(e, loss, wandb_id)  # save checkpoint
            logging.info('Get student validation accuracy')
            self.eval_student(loss)  # get validation metrics
        if e_start >= self.cfg.max_epochs:
            self.eval_student({})

    def fit_contrastive(self, e_start, wandb_id):
        """Perform contrastive distillation

        :param e_start: Start epoch
        :param wandb_id: ID of the wandb run

        Returns:

        """
        for e in range(e_start, self.cfg.max_epochs):
            logging.info(f'Train step {e}')
            loss = self.contrastive_distill()  # perform distillation
            logging.info(f'Completed step {e} with loss {loss}')
            logging.info('Save to checkpoint')
            self.save_to_checkpoint(e, loss, wandb_id)  # save checkpoint
            logging.info('Get student validation accuracy')
            self.eval_student(loss)  # get validation metrics
        if e_start >= self.cfg.max_epochs:
            self.eval_student({})


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    if cfg.search_id != 'None':
        cfg = get_teacher_student_id(cfg, cfg.search_id)  # get teacher and student ids from search id
        # cfg = grid_search(cfg, cfg.search_id)
    tags = [cfg.mode]
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # parse config
    OmegaConf.set_struct(cfg, False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f'Seed: {cfg.seed}')

    # import list of models from the timm library
    models_list = pd.read_csv('files/contdist_model_list.csv')

    # start run timer
    t_start = time.time()
    # get teacher and student model names and parameters from config
    student_name, student_type, student_params = get_model(cfg.student_id, models_list)
    teacher_name, teacher_type, teacher_params = get_model(cfg.teacher_id, models_list)

    # get suitable batch size
    if cfg.optimizer.batch_size == 'auto':
        batch_size = get_batch_size(teacher_name, student_name, device, cfg.loss.name)
        cfg.optimizer.batch_size = batch_size
        config['batch_size'] = batch_size
    cfg = parse_cfg(cfg)

    # initialize the distillation trainer
    trainer = DistillationTrainer(cfg, teacher_name, student_name)
    #if 'imagenet' in cfg.data.dataset:
        # check the accuracies of the teacher and student models
        #trainer.check_accuracies(models_list, (cfg.teacher_id, cfg.student_id))

    st_cfg = {'teacher_name': teacher_name, 'student_name': student_name,
              'ts_diff': trainer.teacher_acc[0] - trainer.student_acc[0],
              'teacher_type': teacher_type, 'student_type': student_type, 'dist_type': f'{teacher_type}>{student_type}',
              'teacher_params': teacher_params, 'student_params': student_params,
              'ts_params_diff': teacher_params - student_params}
    config.update(st_cfg)
    logging.info(f'Run Config: {config}')

    try:
        # load from checkpoint
        logging.info(f'looking for checkpoint at {trainer.checkpoint_path}')
        if not cfg.checkpoint.enabled:
            raise FileNotFoundError
        e_start, loss, wandb_id = trainer.load_from_checkpoint()
        # initialize wandb logger
        wandb.init(id=wandb_id, resume="allow", project=cfg.wandb.project, config=config, tags=tags)
        wandb.run.name = f'{teacher_name}>{student_name}'
        logging.info(f'Loaded from checkpoint: {trainer.checkpoint_path}')
    except (FileNotFoundError, RuntimeError):
        # create checkpoint folder
        os.makedirs(trainer.checkpoint_path, exist_ok=True)
        # initialize wandb logger
        wandb_id = wandb.util.generate_id()
        wandb.init(id=wandb_id, project=cfg.wandb.project, config=config, tags=tags)
        wandb.run.name = f'{teacher_name}>{student_name}'
        e_start = 0

        wandb.log({'teacher_acc': trainer.teacher_acc[0],
                   'student_acc': trainer.student_acc[0],
                   }, step=0)

    try:
        if 'xekl' in cfg.loss.name or 'hinton' in cfg.loss.name:
            trainer.fit_xekl(e_start, wandb_id)  # perform distillation with cross-entropy and KL divergence
        elif cfg.loss.name in ['crd', 'cd']:
            trainer.fit_contrastive(e_start, wandb_id)  # perform contrastive distillation
        else:
            raise Exception(f'Distillation approach {cfg.loss.name} not implemented')

        logging.info(f'Completed distillation for {teacher_name} -> {student_name}')
        wandb.alert(
            title=f'COMPLETED: Distillation Run (seed {cfg.seed})',
            text=f'Completed distillation for {teacher_name} -> {student_name}'
                 f' ({round(time.time() - t_start, 2)}s)',
            level=AlertLevel.INFO
        )
    except AssertionError as e:
        logging.error(f'Distillation for {teacher_name} -> {student_name} failed due to error: {e}')
        wandb.alert(
            title=f'ERROR: Distillation Run (seed {cfg.seed})',
            text=f'Distillation for {teacher_name} -> {student_name} failed due to error: {e}',
            level=AlertLevel.ERROR
        )


if __name__ == '__main__':
    main()
