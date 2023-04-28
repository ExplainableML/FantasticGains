import torch
import logging

import torch.nn.functional as F
import numpy as np
from torchvision import transforms

from sklearn.metrics import accuracy_score
from omegaconf import DictConfig, OmegaConf

from .models import init_timm_model
from .flip_study import get_flips, get_dist_improvement


def get_val_acc(model, loader, cfg, linear=None, theta_slow=None):
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
                out = model(norm_batch(imgs, cfg))
                if linear is not None:
                    out = linear(out)
            _, batch_preds = torch.max(out, 1)
            preds += batch_preds.cpu().tolist()
            true_y += labels.cpu().tolist()
    if theta_slow is not None:
        model.load_state_dict(theta_fast)
    return accuracy_score(true_y, preds) * 100


def get_val_preds(model, loader, cfg, linear=None, theta_slow=None, return_logits=False, return_truey=False):
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
                out = model(norm_batch(imgs, cfg))
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


def get_val_metrics(student, teacher, loader, cfg, zero_preds, linear_s=None, linear_t=None, theta_slow=None, zero_preds_step=None):
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
                out_s = student(norm_batch(imgs, cfg))
                out_t = teacher(norm_batch(imgs, cfg))
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

    metrics = get_metrics(zero_preds, preds, true_y)

    return metrics


def get_metrics(zero_preds, preds, teach_preds, true_y, train=False):
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


def get_batch_size(teacher_name, student_name, device, loss_name, max_batch_size=None, num_iterations=5) -> int:
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
    loss_params = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    loss_params_string = '_'.join([f'{key}_{value}' for key, value in loss_params.items()])
    return loss_params_string


def norm_batch(batch, cfg):
    mean = np.array(cfg['mean'])
    std = np.array(cfg['std'])
    norm = transforms.Normalize(mean=mean, std=std)
    batch = norm(torch.div(batch, 255))
    # for i, img in enumerate(batch):
    #    batch[i] = norm(torch.div(img, 255))
    return batch


def get_flip_masks(out_s, out_t, labels):
    _, s_preds = torch.max(out_s, 1)
    _, t_preds = torch.max(out_t, 1)
    mask_pos = torch.logical_and(torch.eq(t_preds, labels), torch.ne(s_preds, labels))
    mask_neg = torch.logical_and(torch.ne(t_preds, labels), torch.eq(s_preds, labels))
    mask_neut = torch.logical_not(torch.logical_or(mask_pos, mask_neg))

    return mask_pos, mask_neg, mask_neut


def random_search(cfg, search_id=123):
    #runs = [(211, 132), (24, 160), (33, 261), (28, 232), (2, 171), (51, 267)]
    runs = [(77, 242), (95, 88)]
    param_grid = {
        'gamma': [1, 1.5, 2, 5, 10],
    }

    f = 1

    np.random.seed(int(search_id / len(runs)) * f)
    cfg.loss.gamma = float(np.random.choice(param_grid['gamma']))
    cfg.teacher_id = runs[search_id % len(runs)][0]
    cfg.student_id = runs[search_id % len(runs)][1]

    return cfg


def grid_search(cfg, search_id=123):
    runs = [(77, 242), (95, 88)]
    param_grid = {
        'gamma': [1, 1.5, 2, 5, 10],
    }

    grid = []
    for r in runs:
        for gamma in param_grid['gamma']:
            grid.append([r, gamma])
    print(f'Len Grid: {len(grid)}')
    params = grid[search_id]
    cfg.loss.gamma = float(params[1])
    cfg.teacher_id = int(params[0][0])
    cfg.student_id = int(params[0][1])

    return cfg


def contdist_grid_search(cfg, search_id=0):
    students = [261, 160, 132]
    param_grid = {
        'N': [2, 10, 50, 100, 200],
        'tau': [0.99, 0.999, 0.9999]
    }

    grid = []
    for s in students:
        for n in param_grid['N']:
            for t in param_grid['tau']:
                grid.append([s, n, t])
    params = grid[search_id]
    cfg.loss.N = float(params[1])
    cfg.loss.tau = float(params[2])
    cfg.contdist.student_id = int(params[0])

    return cfg


def get_model(id, models_list, return_acc=False):
    name = models_list['modelname'][id]
    type = models_list['modeltype'][id]
    params = models_list['modelparams'][id]
    if return_acc:
        acc = models_list['modeltop1'][id]
        return name, type, params, acc
    else:
        return name, type, params


def get_teacher_student_id(cfg, experiment_id):
    students = [5, 40, 2]
    teachers = [268, 182, 310, 10, 302, 209, 234, 77, 145, 12, 239]

    s_t = []
    for s in students:
        for t in teachers:
            s_t.append([s, t])

    cfg.teacher_id = int(s_t[experiment_id][1])
    cfg.student_id = int(s_t[experiment_id][0])

    return cfg
