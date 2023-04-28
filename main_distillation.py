import os

import hydra
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import wandb
import time
import torch
import logging

import numpy as np
import pandas as pd
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf
from wandb import AlertLevel
import torch.backends.cudnn as cudnn

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


class DistillationTrainer(BaseDisitllationTrainer):
    """Trainer class to perform knowledge distillation from one teacher model to a student model

    :param cfg: Configuration of the distillation run
    :param teacher_name: Name of the teacher model
    :param student_name: Name of the student model
    """

    def __init__(self, cfg, teacher_name, student_name):
        super().__init__(cfg, teacher_name, student_name)
        seed_everything(cfg.seed)

        # initialize the teacher and student models
        if cfg.loss.name in ['crd', 'cd'] or ('mt' in cfg.loss.name and 'adaptive' in cfg.loss.strat):
            student, student_lin, self.cfg_s = init_timm_model(student_name, self.device, split_linear=True)
            teacher, teacher_lin, self.cfg_t = init_timm_model(teacher_name, self.device, split_linear=True)
            s_dim = get_feature_dims(student, self.device)
            t_dim = get_feature_dims(teacher, self.device)
            if 'mt' in cfg.loss.name:
                student_teacher, student_teacher_lin, self.cfg_st = init_timm_model(student_name, self.device,
                                                                                    split_linear=True)

            self.module_list = nn.ModuleList([])
            self.criterion_list = nn.ModuleList([])
            self.module_list.append(student)
            self.module_list.append(student_lin)
            trainable_list = nn.ModuleList([])
            trainable_list.append(student)
            trainable_list.append(student_lin)

            if cfg.loss.name == 'crd':
                criterion_kd = CRDLoss(self.cfg, self.device, s_dim, t_dim)
                self.module_list.append(criterion_kd.embed_s)
                self.module_list.append(criterion_kd.embed_t)
                trainable_list.append(criterion_kd.embed_s)
                trainable_list.append(criterion_kd.embed_t)
            elif 'mt' in cfg.loss.name:
                student_emb = Embed(s_dim, self.cfg.loss.feat_dim)
                teacher_emb = Embed(t_dim, self.cfg.loss.feat_dim)
                student_teacher_emb = Embed(s_dim, self.cfg.loss.feat_dim)
                weighting_params = WeightingParams(self.cfg.loss.feat_dim, self.device)
                self.module_list.append(weighting_params)
                self.module_list.append(student_emb)
                self.module_list.append(student_teacher_emb)
                self.module_list.append(teacher_emb)
                trainable_list.append(weighting_params)
                trainable_list.append(student_emb)
                trainable_list.append(student_teacher_emb)
                trainable_list.append(teacher_emb)
            elif cfg.loss.name == 'cd':
                criterion_kd = SimpleContrastLoss()
            else:
                raise NotImplementedError

            criterion_cls = nn.CrossEntropyLoss()
            criterion_div = DistillKL(self.cfg.loss.kd_T)

            self.criterion_list.append(criterion_cls)  # classification loss
            self.criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
            if cfg.loss.name in ['crd', 'cd']:
                self.criterion_list.append(criterion_kd)  # other knowledge distillation loss

            self.opt = torch.optim.SGD(trainable_list.parameters(), lr=self.cfg.optimizer.lr,
                                       momentum=self.cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)

            if 'mt' in cfg.loss.name:
                self.module_list.append(student_teacher)
                self.module_list.append(student_teacher_lin)
            self.module_list.append(teacher)
            self.module_list.append(teacher_lin)

            if self.device == torch.device("cuda"):
                self.module_list.cuda()
                self.criterion_list.cuda()
                cudnn.benchmark = True

        else:
            self.student, self.cfg_s = init_timm_model(student_name, self.device)
            self.teacher, self.cfg_t = init_timm_model(teacher_name, self.device)
            self.module_list = None
            self.criterion_list = None
            self.opt = torch.optim.SGD(self.student.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum,
                                       weight_decay=cfg.optimizer.weight_decay)

            self.student_teacher, self.cfg_st = init_timm_model(student_name,
                                                                self.device)  # if 'mt' in cfg.loss.name else (None, None)

        # initialize the learning rate scheduler
        if cfg.scheduler.name == 'warmup_cosine':
            self.scheduler = LinearWarmupCosineAnnealingLR(self.opt, warmup_epochs=cfg.scheduler.warmup,
                                                           max_epochs=cfg.max_epochs)
        else:
            self.scheduler = None

        self.theta_slow = self.student.state_dict() if 'mcl' in cfg.loss.name else None

        # initialize the validation and train dataloaders
        self.val_loader = get_ffcv_val_loader(self.cfg_s, self.device, cfg, batch_size=cfg.optimizer.batch_size)
        self.train_loader = get_ffcv_train_loader(self.cfg_s, self.device, cfg)

        if cfg.tag == 'imagenet-subset':
            self.get_imagenet_subset()
            sys.exit()

        # infer the teacher's and student's validation accuracy
        if cfg.loss.name in ['crd', 'cd'] or ('mt' in cfg.loss.name and 'adaptive' in cfg.loss.strat):
            if not 'mt' in cfg.loss.name:
                self.cls_positive, self.cls_negative, self.idx_map, n_data = get_cls_pos_neg(self.train_loader)
                assert n_data == self.cfg.data.n_data, f'Actual dataset size ({n_data}) does not match config ({self.cfg.data.n_data})'
            self.student_acc = [
                get_val_acc(self.module_list[0], self.val_loader, self.cfg_s, linear=self.module_list[1])]
            self.teacher_acc = [
                get_val_acc(self.module_list[-2], self.val_loader, self.cfg_t, linear=self.module_list[-1])]
            self.zero_preds = get_val_preds(self.module_list[0], self.val_loader, self.cfg_s,
                                            linear=self.module_list[1])
        else:
            self.zero_preds = get_val_preds(self.student, self.val_loader, self.cfg_s)
            self.student_acc = [get_val_acc(self.student, self.val_loader, self.cfg_s)]
            self.teacher_acc = [get_val_acc(self.teacher, self.val_loader, self.cfg_t)]
        self.knowledge_gain = [0]
        self.knowledge_loss = [0]

    def xe_kl_distill(self):
        """Perform one epoch of xe-kl knowledge distillation

        Returns: tuple (dist_loss, kl_loss, xe_loss)
            - Distillation loss
            - KL-Divergence loss
            - XE-Classification loss

        """
        self.student.train()
        self.teacher.eval()
        if self.student_teacher is not None:
            self.student_teacher.eval()

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

            # pass batch images through the models
            with torch.cuda.amp.autocast():
                out_s = self.student(norm_batch(imgs, self.cfg_s))
                sp_s = nn.functional.softmax(out_s, dim=1)
                _, s_preds = torch.max(sp_s, 1)
                with torch.no_grad():
                    out_t = self.teacher(norm_batch(imgs, self.cfg_t))
                    sp_t = nn.functional.softmax(out_t, dim=1)
                    _, t_preds = torch.max(sp_t, 1)
                    if self.student_teacher is not None:
                        out_st = self.student_teacher(norm_batch(imgs, self.cfg_st))
                        sp_st = nn.functional.softmax(out_st, dim=1)
                        _, st_preds = torch.max(sp_st, 1)

            top_k, top_k_idx = torch.topk(out_t, self.cfg.loss.k)
            topk_out_s = torch.randn((imgs.size(0), self.cfg.loss.k), device=self.device)
            topk_out_t = torch.randn((imgs.size(0), self.cfg.loss.k), device=self.device)
            topk_out_st = torch.randn((imgs.size(0), self.cfg.loss.k), device=self.device)

            for i, index in enumerate(top_k_idx):
                topk_out_s[i] = torch.index_select(out_s[i], 0, index)
                topk_out_t[i] = torch.index_select(out_t[i], 0, index)
                if self.student_teacher is not None:
                    topk_out_st[i] = torch.index_select(out_st[i], 0, index)

            t_input = 0
            if 'mt' in self.cfg.loss.name:
                pos_mask, neg_mask, neut_mask = get_flip_masks(out_s, out_t, labels)
                t_mask = torch.zeros((imgs.size(0)), device=self.device).to(torch.bool)
                if 'most-conf' in self.cfg.loss.strat:
                    topk_out_mt = torch.randn((imgs.size(0), self.cfg.loss.k), device=self.device)
                    for i in range(imgs.size(0)):
                        if '_u' in self.cfg.loss.strat:
                            t_mask[i] = torch.max(sp_t[i]) > torch.max(sp_st[i])
                        else:
                            t_mask[i] = sp_t[i][labels[i]] > sp_st[i][labels[i]]
                        topk_out_mt[i] = topk_out_t[i] if t_mask[i] else topk_out_st[i]
                    t_input = (torch.sum(t_mask) / imgs.size(0)) * 100
                    st_mask = torch.logical_not(t_mask)
                    if self.cfg.tag == 'gamma-test':
                        loss_t = kl_loss_fn(topk_out_s[t_mask], topk_out_t[t_mask], reduction=False)
                        loss_st = kl_loss_fn(topk_out_s[st_mask], topk_out_st[st_mask], reduction=False) * float(
                            self.cfg.loss.gamma)
                        kl_loss = (torch.sum(loss_t) + torch.sum(loss_st)) / imgs.size(0)
                    else:
                        kl_loss = kl_loss_fn(topk_out_s, topk_out_mt)
                elif 'flip' in self.cfg.loss.strat:
                    topk_out_mt = torch.randn((imgs.size(0), self.cfg.loss.k), device=self.device)
                    if self.cfg.loss.strat == 'flip-avg':
                        for i in range(imgs.size(0)):
                            if neut_mask[i]:
                                topk_out_mt[i] = torch.mean(torch.stack((topk_out_t[i], topk_out_s[i]), dim=0), 0)
                            else:
                                topk_out_mt[i] = topk_out_t[i] if pos_mask[i] else topk_out_st[i]
                    elif self.cfg.loss.strat == 'flip-st':
                        for i in range(imgs.size(0)):
                            topk_out_mt[i] = topk_out_t[i] if pos_mask[i] else topk_out_st[i]
                        t_input = (torch.sum(pos_mask).item() / imgs.size(0)) * 100
                    elif self.cfg.loss.strat == 'flip-t':
                        for i in range(imgs.size(0)):
                            topk_out_mt[i] = topk_out_st[i] if neg_mask[i] else topk_out_t[i]
                        t_input = (1 - torch.sum(neg_mask).item() / imgs.size(0)) * 100
                    elif self.cfg.loss.strat == 'self-dist':
                        topk_out_mt = topk_out_st
                    kl_loss = kl_loss_fn(topk_out_s, topk_out_mt)
                else:
                    raise NotImplementedError

            else:
                pos_mask, neg_mask, neut_mask = get_flip_masks(out_s, out_t, labels)
                kl_loss = torch.zeros(1, device=self.device)
                if self.cfg.on_flip.pos == 'distill' and self.cfg.on_flip.neg == 'distill' and self.cfg.on_flip.neut == 'distill':
                    kl_loss = kl_loss_fn(topk_out_s, topk_out_t)
                elif self.cfg.on_flip.pos == 'distill' and self.cfg.on_flip.neut == 'distill':
                    kl_loss = kl_loss_fn(topk_out_s[torch.logical_not(neg_mask)],
                                         topk_out_t[torch.logical_not(neg_mask)])
                elif self.cfg.on_flip.pos == 'distill':
                    kl_loss = kl_loss_fn(topk_out_s[pos_mask], topk_out_t[pos_mask])
                if torch.isnan(kl_loss):
                    kl_loss = torch.zeros(1, device=self.device)

            xe_loss = xe_loss_fn(sp_s, labels)
            loss = float(self.cfg.loss.alpha) * kl_loss + (1 - float(self.cfg.loss.alpha)) * xe_loss

            loss.backward()
            self.opt.step()

            if self.theta_slow is not None and j % self.cfg.loss.N == 0:
                theta = self.student.state_dict()
                for key in theta.keys():
                    self.theta_slow[key] = self.cfg.loss.tau * self.theta_slow[key] + (1 - self.cfg.loss.tau) * theta[
                        key]

            train_metrics = get_metrics(st_preds.cpu().tolist(), s_preds.cpu().tolist(), t_preds.cpu().tolist(), labels.cpu().tolist(), train=True)

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
            if 'mt' in self.cfg.loss.name:
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

    def adaptive_mt_distill(self):
        """Perform one epoch of contrastive distillation

        Returns: tuple (dist_loss, cls_loss, div_loss, kd_loss)
            - Distillation loss
            - Classification loss
            - KL-divergence loss
            - Contrastive loss
        """
        # set modules as train()
        for module in self.module_list:
            module.train()
        # set teacher and student-teacher as eval()
        for i in range(6, 10):
            self.module_list[i].eval()

        xe_loss_fn = self.criterion_list[0]
        kl_loss_fn = self.criterion_list[1]

        model_s = self.module_list[0]
        model_s_lin = self.module_list[1]
        model_s_emb = self.module_list[3]
        model_st = self.module_list[-4]
        model_st_lin = self.module_list[-3]
        model_st_emb = self.module_list[4]
        model_t = self.module_list[-2]
        model_t_lin = self.module_list[-1]
        model_t_emb = self.module_list[5]
        weighting_params = self.module_list[2]

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

        for imgs, labels, index in self.train_loader:
            with torch.cuda.amp.autocast():
                out_s = model_s(norm_batch(imgs, self.cfg_s))
                logit_s = model_s_lin(out_s)
                feat_s = model_s_emb(out_s)
                with torch.no_grad():
                    out_t = model_t(norm_batch(imgs, self.cfg_t))
                    logit_t = model_t_lin(out_t)
                    feat_t = model_t_emb(out_t)
                    out_st = model_st(norm_batch(imgs, self.cfg_st))
                    logit_st = model_st_lin(out_st)
                    feat_st = model_st_emb(out_st)

            sp_s = nn.functional.softmax(logit_s, dim=1)
            _, s_preds = torch.max(sp_s, 1)
            sp_st = nn.functional.softmax(logit_st, dim=1)
            _, st_preds = torch.max(sp_st, 1)

            teacher_weighting = torch.zeros((imgs.size(0), 2), device=self.device)
            for i in range(imgs.size(0)):
                teacher_weighting[i][0] = torch.dot(weighting_params.params, torch.mul(feat_st[i], feat_s[i]))
                teacher_weighting[i][1] = torch.dot(weighting_params.params, torch.mul(feat_t[i], feat_s[i]))

            _, t_mask = torch.max(teacher_weighting, 1)
            t_mask = t_mask.to(torch.bool)
            t_input = torch.sum(t_mask) / imgs.size(0) * 100

            st_mask = torch.logical_not(t_mask)

            loss_t = kl_loss_fn(logit_s[t_mask], logit_t[t_mask], reduction=False)
            loss_st = kl_loss_fn(logit_s[st_mask], logit_st[st_mask], reduction=False) * float(self.cfg.loss.gamma)
            kl_loss = (torch.sum(loss_t) + torch.sum(loss_st)) / imgs.size(0)

            xe_loss = xe_loss_fn(sp_s, labels)
            loss = float(self.cfg.loss.alpha) * kl_loss + (1 - float(self.cfg.loss.alpha)) * xe_loss

            loss.backward()
            self.opt.step()

            train_metrics = get_metrics(st_preds.cpu().tolist(), s_preds.cpu().tolist(), labels.cpu().tolist())
            pos_mask, neg_mask, neut_mask = get_flip_masks(logit_s, logit_t, labels)

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
            neg_to_teacher.update(torch.sum(t_mask[neg_mask] / torch.sum(neg_mask)), imgs.size(0))
            pos_to_teacher.update(torch.sum(t_mask[pos_mask] / torch.sum(pos_mask)), imgs.size(0))

        if self.scheduler is not None:
            self.scheduler.step()

        out = {'dist_loss': avg_loss.avg, 'kl_loss': avg_kl_loss.avg, 'xe_loss': avg_xe_loss.avg,
               'teacher_input': teacher_input.avg, 'pos_to_teacher': pos_to_teacher.avg,
               'neg_to_teacher': neg_to_teacher.avg,
               'k_gain_train': train_k_gain.avg, 'k_loss_train': train_k_loss.avg,
               'dist_delta_train': train_dist_delta.avg,
               'pos_flips': pos_flips.avg, 'neg_flips': neg_flips.avg, 'neut_flips': neut_flips.avg}

        return out

    def contrastive_ditsill(self):
        """Perform one epoch of contrastive distillation

        Returns: tuple (dist_loss, cls_loss, div_loss, kd_loss)
            - Distillation loss
            - Classification loss
            - KL-divergence loss
            - Contrastive loss
        """
        # set modules as train()
        for module in self.module_list:
            module.train()
        # set teacher as eval()
        self.module_list[-1].eval()

        criterion_cls = self.criterion_list[0]
        criterion_div = self.criterion_list[1]
        criterion_kd = self.criterion_list[2]

        model_s = self.module_list[0]
        model_s_lin = self.module_list[1]
        model_t = self.module_list[-2]
        model_t_lin = self.module_list[-1]

        losses = AverageMeter()
        losses_cls = AverageMeter()
        losses_div = AverageMeter()
        losses_kd = AverageMeter()

        for input, target, index in self.train_loader:
            if self.cfg.loss.name == 'crd':
                index = torch.tensor([self.idx_map[idx] for idx in index.cpu().tolist()], device=self.device)
                contrast_idx = get_contrast_idx(index, target, self.cls_negative, self.cfg.loss.nce_k)
                contrast_idx = torch.tensor(contrast_idx, device=self.device).int()

            with torch.cuda.amp.autocast():
                feat_s = model_s(norm_batch(input, self.cfg_s))
                logit_s = model_s_lin(feat_s)
                with torch.no_grad():
                    feat_t = model_t(norm_batch(input, self.cfg_t))
                    logit_t = model_t_lin(feat_t)
                    feat_t = feat_t.detach()

            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)

            f_s = feat_s.float()
            f_t = feat_t.float()

            if self.cfg.loss.name == 'crd':
                loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
            else:
                loss_kd = criterion_kd(f_s, f_t)

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
        if self.cfg.loss.name in ['crd', 'cd'] or ('mt' in self.cfg.loss.name and 'adaptive' in self.cfg.loss.strat):
            s_metrics = get_val_metrics(self.module_list[0], self.module_list[-2], self.val_loader, self.cfg_s, self.zero_preds,
                                        linear_s=self.module_list[1], linear_t=self.module_list[-1])
        else:
            s_metrics = get_val_metrics(self.student, self.teacher, self.val_loader, self.cfg_s, self.zero_preds,
                                        theta_slow=self.theta_slow)
        logging.info(f'Metrics: {s_metrics}')
        # t_acc = get_val_acc(self.teacher, self.val_loader, self.cfg_t)
        t_acc = self.teacher_acc[-1]
        self.student_acc.append(s_metrics['student_acc'])
        self.teacher_acc.append(t_acc)
        stats = {'lr': self.scheduler.get_lr()[0] if self.scheduler is not None else self.cfg.optimizer.lr,
                 'student_acc': s_metrics['student_acc'],
                 'teacher_acc': self.teacher_acc[-1],
                 'dist_delta': s_metrics['student_acc'] - self.student_acc[0],
                 'knowledge_gain': s_metrics['knowledge_gain'],
                 'knowledge_loss': s_metrics['knowledge_loss']}
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
            if 'mt' in self.cfg.loss.name and self.cfg.loss.strat == 'adaptive':
                loss = self.adaptive_mt_distill()
            else:
                loss = self.xe_kl_distill()
            logging.info('Save to checkpoint')
            self.save_to_checkpoint(e, loss, wandb_id)
            logging.info('Get student validation accuracy')
            self.eval_student(loss)
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
            loss = self.contrastive_ditsill()
            logging.info(f'Completed step {e} with loss {loss}')
            if self.cfg.checkpoint.enabled:
                logging.info('Save to checkpoint')
                self.save_to_checkpoint(e, loss, wandb_id)
            logging.info('Get student validation accuracy')
            self.eval_student(loss)
        if e_start >= self.cfg.max_epochs:
            self.eval_student({'dist_loss': 0, 'kl_loss': 0, 'xe_loss': 0, 'teacher_input': None})

    def ensemble_baseline(self):
        s_metrics = get_ensemble_metrics(self.student, self.teacher, self.val_loader, self.cfg_s, self.cfg_t,
                                    self.zero_preds)

        t_acc = self.teacher_acc[-1]
        self.student_acc.append(s_metrics['acc'])
        self.teacher_acc.append(t_acc)
        log = {'lr': self.scheduler.get_lr()[0] if self.scheduler is not None else self.cfg.optimizer.lr,
               'student_acc': s_metrics['acc'],
               'teacher_acc': self.teacher_acc[-1],
               'dist_delta': s_metrics['acc'] - self.student_acc[0],
               'knowledge_gain': s_metrics['knowledge_gain'],
               'knowledge_loss': s_metrics['knowledge_loss']}
        logging.info(f'Log stats: {log}')
        wandb.log(log)


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    if cfg.search_id != 'None':
        cfg = get_teacher_student_id(cfg, cfg.search_id)
        #cfg = grid_search(cfg, cfg.search_id)
    tags = [cfg.tag]
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
    # use cfg seed to draw two random models from the filtered model list
    student_name, student_type, student_params = get_model(cfg.student_id, models_list)
    teacher_name, teacher_type, teacher_params = get_model(cfg.teacher_id, models_list)

    if cfg.optimizer.batch_size == 'auto':
        batch_size = get_batch_size(teacher_name, student_name, device, cfg.loss.name)
        cfg.optimizer.batch_size = batch_size
        config['batch_size'] = batch_size
    cfg = parse_cfg(cfg)
    trainer = DistillationTrainer(cfg, teacher_name, student_name)
    trainer.check_accuracies(models_list, (cfg.teacher_id, cfg.student_id))

    st_cfg = {'teacher_name': teacher_name, 'student_name': student_name,
              'ts_diff': trainer.teacher_acc[0] - trainer.student_acc[0],
              'teacher_type': teacher_type, 'student_type': student_type, 'dist_type': f'{teacher_type}>{student_type}',
              'teacher_params': teacher_params, 'student_params': student_params,
              'ts_params_diff': teacher_params - student_params}
    config.update(st_cfg)
    logging.info(f'Run Config: {config}')

    try:
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
            trainer.fit_xekl(e_start, wandb_id)
        elif cfg.loss.name in ['crd', 'cd']:
            trainer.fit_contrastive(e_start, wandb_id)
        elif cfg.loss.name == 'ensemble_baseline':
            trainer.ensemble_baseline()
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
