"""
This script is licensed under the MIT License.
For more details, see the LICENSE file in the root directory of this repository.

(c) 2024 Lukas Thede
"""

import os
import torch
import logging
import math

import numpy as np
from torchvision import transforms

from sklearn.metrics import accuracy_score
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import _LRScheduler

from .models import init_timm_model
from .flip_study import get_flips, get_dist_improvement


def get_val_acc(model, loader, cfg, linear=None, theta_slow=None, norm=True):
    """Get validation accuracy of a model

    :param model: model to evaluate
    :param loader: validation loader
    :param cfg: config
    :param linear: linear layer to apply to model output
    :param theta_slow: slow weights to load into model
    :param norm: Normalize batch inputs

    :Returns: validation accuracy
    """
    preds = []
    true_y = []
    model.eval()
    if linear is not None:
        linear.eval()
    if theta_slow is not None:
        theta_fast = model.state_dict()
        model.load_state_dict(theta_slow)
    for imgs, labels, idxs in loader:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                out = model(norm_batch(imgs, cfg) if norm else imgs)
                if linear is not None:
                    out = linear(out)
            _, batch_preds = torch.max(out, 1)
            preds += batch_preds.cpu().tolist()
            true_y += labels.cpu().tolist()
    if theta_slow is not None:
        model.load_state_dict(theta_fast)
    return accuracy_score(true_y, preds) * 100


def get_val_preds(model, loader, cfg, linear=None, theta_slow=None, return_logits=False, return_truey=False, norm=True):
    """Get validation predictions of a model

    :param model: model to evaluate
    :param loader: validation loader
    :param cfg: config
    :param linear: linear layer to apply to model output
    :param theta_slow: slow weights to load into model
    :param return_logits: whether to return logits
    :param return_truey: whether to return true labels
    :param norm: Norm batch inputs

    :Returns: validation predictions
    """
    preds = []
    true_y = []
    model.eval()
    if linear is not None:
        linear.eval()
    if theta_slow is not None:
        theta_fast = model.state_dict()
        model.load_state_dict(theta_slow)
    for imgs, labels, idxs in loader:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                out = model(norm_batch(imgs, cfg) if norm else imgs)
                if linear is not None:
                    out = linear(out)
            if return_logits:
                batch_preds = torch.nn.functional.softmax(out, dim=1)
            else:
                _, batch_preds = torch.max(out, 1)
            preds += batch_preds.cpu().tolist()
            true_y += labels.cpu().tolist()
    if theta_slow is not None:
        model.load_state_dict(theta_fast)
    if return_truey:
        return preds, true_y
    else:
        return preds


def get_val_metrics(student, teacher, loader, cfg, zero_preds, linear_s=None, linear_t=None, theta_slow=None, zero_preds_step=None, norm=True):
    """Get validation metrics of a student and teacher model

    :param student: student model
    :param teacher: teacher model
    :param loader: validation loader
    :param cfg: config
    :param zero_preds: zero predictions of the student model
    :param linear_s: linear layer to apply to student model output
    :param linear_t: linear layer to apply to teacher model output
    :param theta_slow: slow weights to load into student model
    :param zero_preds_step: zero predictions of the student model of the previous step (for sequential distillation)
    :param norm: Norm batch inputs

    :Returns: validation metrics
    """
    preds_s = []
    preds_t = []
    true_y = []
    student.eval()
    teacher.eval()
    if linear_s is not None:
        linear_s.eval()
        linear_t.eval()
    if theta_slow is not None:
        theta_fast = student.state_dict()
        student.load_state_dict(theta_slow)
    for imgs, labels, idxs in loader:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                out_s = student(norm_batch(imgs, cfg) if norm else imgs)
                out_t = teacher(norm_batch(imgs, cfg) if norm else imgs)
                if linear_s is not None:
                    out_s = linear_s(out_s)
                    out_t = linear_t(out_t)
            _, batch_preds_s = torch.max(out_s, 1)
            preds_s += batch_preds_s.cpu().tolist()
            _, batch_preds_t = torch.max(out_t, 1)
            preds_t += batch_preds_t.cpu().tolist()
            true_y += labels.cpu().tolist()
    if theta_slow is not None:
        student.load_state_dict(theta_fast)

    metrics = get_metrics(zero_preds, preds_s, preds_t, true_y)
    if zero_preds_step is not None:
        metrics_step = get_metrics(zero_preds_step, preds_s, preds_t, true_y)
        for key in metrics_step.keys():
            metrics[f'{key}_step'] = metrics_step[key]
    return metrics


def get_ensemble_metrics(model_a, model_b, loader, cfg_a, cfg_b, zero_preds):
    """Get validation metrics of an ensemble of two models

    :param model_a: first model
    :param model_b: second model
    :param loader: validation loader
    :param cfg_a: config of first model
    :param cfg_b: config of second model
    :param zero_preds: zero predictions of the second model

    :Returns: validation metrics
    """
    preds = []
    true_y = []
    model_a.eval()
    model_b.eval()

    for imgs, labels, idxs in loader:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                out_a = model_a(norm_batch(imgs, cfg_a))
                out_b = model_b(norm_batch(imgs, cfg_b))
            out_a = torch.nn.functional.softmax(out_a, dim=1)
            out_b = torch.nn.functional.softmax(out_b, dim=1)
            _, batch_preds_a = torch.max(out_a, 1)
            _, batch_preds_b = torch.max(out_b, 1)

            batch_preds = []
            for i in range(imgs.size(0)):
                if torch.max(out_a[i]) > torch.max(out_b[i]):
                    batch_preds.append(batch_preds_a[i].item())
                else:
                    batch_preds.append(batch_preds_b[i].item())
            true_y += labels.cpu().tolist()
            preds += batch_preds

    metrics = get_metrics(zero_preds, preds, preds, true_y)

    return metrics


def get_metrics(zero_preds, preds, teach_preds, true_y, train=False):
    """Get validation metrics of a student and teacher model

    :param zero_preds: zero predictions of the student model
    :param preds: predictions of the student model
    :param teach_preds: predictions of the teacher model
    :param true_y: ground truth labels
    :param train: whether the metrics are for training or validation

    :Returns: validation metrics
    """
    acc_s = accuracy_score(true_y, preds) * 100
    acc_st = accuracy_score(true_y, zero_preds) * 100
    gain_loss = get_flips(zero_preds, preds, true_y)
    zero_flips = get_flips(zero_preds, teach_preds, true_y)
    pred_flips = get_flips(preds, teach_preds, true_y)
    improvement = get_dist_improvement(preds, zero_preds, teach_preds, true_y) if not train else {}
    metrics = {'student_acc': acc_s,
               'dist_delta': acc_s - acc_st,
               'knowledge_gain': gain_loss['pos_rel'],
               'knowledge_loss': gain_loss['neg_rel'],
               'positive_flips': pred_flips['pos_rel'],
               'negative_flips': pred_flips['neg_rel'],
               'pos_flips_delta': pred_flips['pos_rel']-zero_flips['pos_rel'],
               'neg_flips_delta': pred_flips['neg_rel']-zero_flips['neg_rel']}
    metrics = {**metrics, **improvement}
    return metrics


def get_batch_size(teacher_name, student_name, device, loss_name, max_batch_size=None, num_iterations=5):
    """Get the maximum batch size for a given teacher and student model

    :param teacher_name: name of the teacher model
    :param student_name: name of the student model
    :param device: device to run the models on
    :param loss_name: name of the loss function
    :param max_batch_size: maximum batch size to test
    :param num_iterations: number of iterations to test

    :Returns: maximum batch size
    """
    logging.info(f't: {teacher_name} - s: {student_name}')
    student, _ = init_timm_model(student_name, device)
    teacher, _ = init_timm_model(teacher_name, device)
    student.train()
    teacher.eval()
    if 'kl' in loss_name:
        student_teacher, _ = init_timm_model(student_name, device)
        student_teacher.eval()
    input_shape = (3, 224, 224)
    output_shape = (1000,)
    optimizer = torch.optim.Adam(student.parameters())
    kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    xe_loss_fn = torch.nn.CrossEntropyLoss()
    logging.info('Find max batch size')
    batch_size = 2
    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break
        try:
            for _ in range(num_iterations):
                # dummy inputs and targets
                inputs = torch.rand(*(batch_size, *input_shape), device=device, dtype=torch.half)
                targets = torch.rand(*(batch_size, *output_shape), device=device, dtype=torch.half)
                with torch.cuda.amp.autocast():
                    s_outputs = student(inputs)
                    with torch.no_grad():
                        t_outputs = teacher(inputs)
                        if 'mt' in loss_name:
                            st_outputs = student_teacher(inputs)
                topk_out_s = torch.randn((inputs.size(0), 1000), device=device)
                topk_out_t = torch.randn((inputs.size(0), 1000), device=device)
                masks = [torch.randn((inputs.size(0)), device=device) for i in range(3)]
                puffer = torch.rand(*(batch_size, *input_shape), device=device, dtype=torch.half)
                xe_loss = xe_loss_fn(targets, s_outputs)
                kl_loss = kl_loss_fn(t_outputs, s_outputs)
                loss = xe_loss + kl_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            batch_size *= 2
        except RuntimeError as e:
            batch_size //= 2
            break
    del optimizer, student, teacher
    torch.cuda.empty_cache()
    crash_list = ['mixer_b16_224', 'xcit_nano_12_p16_224_dist', 'vitcon_base_patch8_224', 'resmlp_12_224',
                  'resmlp_24_224', 'resmlp_24_distilled_224', 'wide_resnet50_2', 'xcit_small_24_p16_224', 'vit_small_patch32_224']
    if 'cd' in loss_name or 'crd' in loss_name or student_name in crash_list:
        batch_size //= 2
    logging.info(f'Set batch size to {batch_size}')
    return batch_size


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def param_string(cfg):
    """Create a string from the parameters of a config file

    :param cfg: config file

    :Returns: string of parameters
    """
    loss_params = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    loss_params_string = '_'.join([f'{key}_{value}' for key, value in loss_params.items()])
    return loss_params_string


def norm_batch(batch, cfg):
    """Normalize a batch of images

    :param batch: batch of images
    :param cfg: model config

    :Returns: normalized batch
    """
    mean = np.array(cfg['mean'])
    std = np.array(cfg['std'])
    norm = transforms.Normalize(mean=mean, std=std)
    batch = norm(torch.div(batch, 255))
    return batch


def get_flip_masks(out_s, out_t, labels):
    """Get the flip masks for a batch of images

    :param out_s: output of the student model
    :param out_t: output of the teacher model
    :param labels: labels of the batch

    :Returns: flip masks
    """
    _, s_preds = torch.max(out_s, 1)
    _, t_preds = torch.max(out_t, 1)
    mask_pos = torch.logical_and(torch.eq(t_preds, labels), torch.ne(s_preds, labels))
    mask_neg = torch.logical_and(torch.ne(t_preds, labels), torch.eq(s_preds, labels))
    mask_neut = torch.logical_not(torch.logical_or(mask_pos, mask_neg))

    return mask_pos, mask_neg, mask_neut


def get_model(id, models_list, return_acc=False):
    """Get the model name, type and parameters from the models list

    :param id: id of the model
    :param models_list: list of models
    :param return_acc: whether to return the top1 accuracy of the model

    :Returns: model name, type, parameters and optionally top1 accuracy
    """
    name = models_list['modelname'][id]
    type = models_list['modeltype'][id]
    params = models_list['modelparams'][id]
    if return_acc:
        acc = models_list['modeltop1'][id]
        return name, type, params, acc
    else:
        return name, type, params


def get_teacher_student_id(cfg, experiment_id):
    """Get the teacher and student id from the experiment id

    :param cfg: experiment config
    :param experiment_id: experiment id

    :Returns: model config with teacher and student id
    """
    students = [41, 5, 26, 33, 131, 132, 48, 214, 2, 77]
    teachers = [268, 234, 302, 209, 10, 182, 310, 12, 239, 145]

    s_t = []
    for s in students:
        for t in teachers:
            s_t.append([s, t])

    cfg.teacher_id = int(s_t[experiment_id][1])
    cfg.student_id = int(s_t[experiment_id][0])

    return cfg


def soup_student_weights(student, teachers, cfg):
    """Average (soup) the weights of student models distilled with different teacher models

    :param student: student model
    :param teachers: list of teacher models
    :param cfg: experiment config

    :Returns: souped student model weights
    """
    logging.info(f'Souping the weights of {student} distilled with {teachers}')
    state_dicts = []
    for teacher in teachers:
        path = os.path.join(
            cfg.checkpoint.dir, cfg.data.dataset + f'_ffcv_{cfg.ffcv_dtype}', cfg.ffcv_augmentation, cfg.loss.name,
            param_string(cfg.loss), f'{teacher}>{student}', f'lr_{cfg.optimizer.lr}',
            f'batch_size_{cfg.optimizer.batch_size}', param_string(cfg.scheduler), f'seed_{cfg.seed}',
            param_string(cfg.on_flip))
        logging.info(f'Looking for checkpoint at {path}')
        ckp = torch.load(path + '/student_checkpoint.pt')
        state_dicts.append(ckp['model_state_dict'])
        logging.info(f'Loaded checkpoint for {teacher}')

    soup_state_dict = {}
    for key in state_dicts[0].keys():
        weights = torch.randn([len(teachers)]+list(state_dicts[0][key].shape))
        for i in range(len(teachers)):
            weights[i] = state_dicts[i][key]
        soup_state_dict[key] = torch.mean(weights, dim=0)
    return soup_state_dict


def parse_cfg(cfg: DictConfig):
    """Parse the config file

    :param cfg: config file

    :Returns: parsed config file
    """
    # adjust lr according to batch size
    cfg.num_nodes = OmegaConf.select(cfg, "num_nodes", default=1)

    if cfg.strategy == 'ddp':
        scale_factor = cfg.optimizer.batch_size * len(cfg.devices) * cfg.num_nodes / 256
    else:
        scale_factor = cfg.optimizer.batch_size / 256

    cfg.optimizer.lr = cfg.optimizer.lr * scale_factor

    if 'mcl' in cfg.loss.name:
        cfg.loss.N = cfg.loss.N / scale_factor

    return cfg


class CosineAnnealingLRWarmup(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1, warmup_iters=0, min_lr=0):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
        self.min_lr = min_lr

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [base_lr * self.last_epoch / (self.warmup_iters + 1e-8) for base_lr in self.base_lrs]
        else:
            return [self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(
                math.pi * (self.last_epoch - self.warmup_iters) / (self.total_iters - self.warmup_iters))) for base_lr
                    in self.base_lrs]


def load_pretrain_weights(model, path, strict=True, drop_linear=False):
    checkpoint = torch.load(path)
    if drop_linear:
        checkpoint['model_state_dict'] = {k: v for k, v in checkpoint['model_state_dict'].items() if 'fc' not in k and 'classifier' not in k and 'head' not in k}
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    logging.info(f'Loaded pre-trained weights from {path}')
    return model


def label_smoothing(p, alpha=0.1):
    u = torch.ones_like(p) / p.size(1)
    return (1 - alpha) * p + alpha * u
