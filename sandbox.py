import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from distillation.contrastive.CRDloss import CRDLoss, DistillKL


def mask(device='cuda'):
    batch_size = 128

    out_s = torch.rand((batch_size, 2), device=device)
    out_t = torch.rand((batch_size, 2), device=device)
    labels = torch.ones(batch_size, device=device)

    _, s_preds = torch.max(out_s, 1)
    _, t_preds = torch.max(out_t, 1)
    mask_pos = torch.logical_and(torch.eq(s_preds, labels), torch.ne(t_preds, labels))
    mask_neg = torch.logical_and(torch.ne(s_preds, labels), torch.eq(t_preds, labels))
    mask_neut = torch.logical_not(torch.logical_or(mask_pos, mask_neg))
    mask_pos_neut = torch.logical_not(mask_neg)
    out_s_pos = out_s[mask_pos]

    kl_loss_fn = DistillKL(0.1)
    loss_target = kl_loss_fn(out_s[mask_pos_neut], out_t[mask_pos_neut])
    loss_pos_neut = kl_loss_fn(out_s[mask_pos_neut], out_t[mask_pos_neut], reduction=False)
    loss_pos = kl_loss_fn(out_s[mask_pos], out_t[mask_pos], reduction=False)
    loss_neut = kl_loss_fn(out_s[mask_neut], out_t[mask_neut], reduction=False)
    loss_combined = (loss_pos + loss_neut) / torch.sum(mask_pos_neut)
    test


def mt_out(device='cuda'):
    batch_size = 128

    out_s = torch.rand((batch_size, 5), device=device)
    out_t = torch.rand((batch_size, 5), device=device)*100
    labels = torch.randint(0, 4, [batch_size], device=device)

    out_ts = torch.zeros((batch_size, 5), device=device)

    out_s = nn.functional.softmax(out_s, dim=1)
    out_t = nn.functional.softmax(out_t, dim=1)

    for i in range(batch_size):
        out_ts[i] = out_t[i] if out_t[i][labels[i]] > out_s[i][labels[i]] else out_s[i]
        print(f'Image {i} - Label {labels[i].item()}')
        print(f'Teacher: {out_t[i]}')
        print(f'Student: {out_s[i]}')
        print(f'combined: {out_ts[i]}')
    test


def agreement():
    batch_size = 128

    out_s = torch.rand((batch_size, 2), device='cuda')
    out_t = torch.rand((batch_size, 2), device='cuda')
    _, s_preds = torch.max(out_s, 1)
    _, t_preds = torch.max(out_t, 1)

    agree = torch.div(torch.sum(torch.eq(t_preds, s_preds)), batch_size)
    test


def logit_weight():
    batch_size = 128

    out_s = torch.rand((batch_size, 1000), device='cuda')
    out_t = torch.rand((batch_size, 1000), device='cuda')
    out_s = nn.functional.log_softmax(out_s, dim=1)
    out_t = nn.functional.log_softmax(out_t, dim=1)

    top_10, top_10_idx = torch.topk(out_t, 10)
    top10_out_s = torch.randn((batch_size, 10), device='cuda')
    top10_out_t = torch.randn((batch_size, 10), device='cuda')
    for i, index in enumerate(top_10_idx):
        top10_out_s[i] = torch.index_select(out_s[i], 0, index)
        top10_out_t[i] = torch.index_select(out_t[i], 0, index)

    logit_sum_s = torch.sum(out_s, dim=1)
    t10logit_sum_s = torch.sum(top10_out_s, dim=1)
    t10weight_s = torch.mean(torch.div(torch.sum(top10_out_s, dim=1), torch.sum(out_s, dim=1)))

    test


def random_search(cfg, id=123):
    #runs = [(211, 132), (24, 160), (33, 261), (28, 232), (2, 171), (51, 267)]
    runs = [(33, 261), (28, 232), (51, 267)]
    param_grid = {
        'lr': [1e-2, 1e-3],
        'alpha': [0, 1],
        'k': [10, 100, 1000],
        'kd_T': [0.1, 1, 10]
    }

    f = 1

    np.random.seed(int(id / len(runs)) * f)
    cfg.optimizer.lr = float(np.random.choice(param_grid['lr']))
    cfg.loss.alpha = round(float(np.random.uniform(param_grid['alpha'][0], param_grid['alpha'][1])), 2)
    cfg.loss.k = float(np.random.choice(param_grid['k']))
    cfg.loss.kd_T = float(np.random.choice(param_grid['kd_T']))
    cfg.teacher_id = runs[id % len(runs)][0]
    cfg.student_id = runs[id % len(runs)][1]

    return cfg


def grid_search(cfg, id=123):
    runs = [397, 235, 63, 197, 105, 92]
    param_grid = {
        'lr': [1e-2, 1e-3, 1e-4, 1e-5],
        'tau': [0.9, 0.99, 0.999, 0.9999],
        'N': [2, 10, 50, 100],
        'dist_setting': [{'alpha': 0.75, 'k': 30, 'kd_T': 4},
                         {'alpha': 0.75, 'k': 30, 'kd_T': 0.1},
                         {'alpha': 0.5, 'k': 30, 'kd_T': 0.1}]
    }

    grid = []
    for r in runs:
        for lr in param_grid['lr']:
            for tau in param_grid['tau']:
                for n in param_grid['N']:
                    for d_s in param_grid['dist_setting']:
                        grid.append([r, lr, tau, n, d_s])
    print(f'Len Grid: {len(grid)}')
    params = grid[id]
    cfg.seed = params[0]
    cfg.optimizer.lr = params[1]
    cfg.loss.tau = params[2]
    cfg.loss.N = params[3]
    cfg.loss.alpha = params[4]['alpha']
    cfg.loss.k = params[4]['k']
    cfg.loss.kd_T = params[4]['kd_T']

    return cfg


def random_search_test():
    cfg = OmegaConf.create({'optimizer': {'lr': 0},
                            'loss': {'tau': 0, 'alpha': 0, 'k': 0, 'kd_T': 0, 'N': 0},
                            'student_id': 0, 'teacher_id': 0})
    param_combinations = []
    for i in range(10):
        new_cfg = random_search(cfg, id=i)
        print(f'{i}: {new_cfg}')
        param_combinations.append(
            [new_cfg.optimizer.lr, new_cfg.loss.tau, new_cfg.loss.N, new_cfg.loss.alpha, new_cfg.loss.k,
             new_cfg.loss.kd_T, new_cfg.teacher_id, new_cfg.student_id])
    print(f'unique combinations: {len(np.unique(param_combinations, axis=0)) / 6}')
    print(f'evaluated runs: {len(np.unique(param_combinations, axis=0))}')


if __name__ == "__main__":
    mt_out('cpu')
