import torch
import torch.nn as nn
import json
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
    mask_pos = torch.logical_and(torch.eq(t_preds, labels), torch.ne(s_preds, labels))
    mask_neg = torch.logical_and(torch.ne(t_preds, labels), torch.eq(s_preds, labels))
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


def sim_model_output(acc, labels, classes):
    batch_size = labels.size(0)
    out = torch.rand((batch_size, 5))
    for i in range(batch_size):
        for c in range(classes):
            if np.random.uniform(0, 1) < acc:
                pred = labels[i]
            else:
                errors = [c for c in range(classes) if c != labels[i]]
                pred = np.random.choice(errors, 1)
            if c == pred:
                out[i][c] = np.random.uniform(0.75, 1)
            else:
                out[i][c] = np.random.uniform(0, 0.7)
    return out


def mt_out(device='cpu'):
    batch_size = 128
    kl_loss_fn = DistillKL(1)
    gamma = 1

    labels = torch.randint(0, 4, [batch_size], device=device)
    out_s = sim_model_output(0.8, labels, 5).to(device)
    out_st = sim_model_output(0.7, labels, 5).to(device)
    out_t = sim_model_output(0.9, labels, 5).to(device)

    sp_s = nn.functional.log_softmax(out_s, dim=1)
    sp_st = nn.functional.softmax(out_st, dim=1)
    sp_t = nn.functional.softmax(out_t, dim=1)

    _, s_preds = torch.max(out_s, 1)
    _, t_preds = torch.max(out_t, 1)
    mask_pos = torch.logical_and(torch.eq(t_preds, labels), torch.ne(s_preds, labels))
    mask_neg = torch.logical_and(torch.ne(t_preds, labels), torch.eq(s_preds, labels))
    mask_neut = torch.logical_not(torch.logical_or(mask_pos, mask_neg))

    ts_most_conf = torch.zeros((batch_size, 5), device=device)
    ts_flip_avg = torch.zeros((batch_size, 5), device=device)
    t_mask = torch.zeros((batch_size), device=device).to(torch.bool)

    for i in range(batch_size):
        t_mask[i] = torch.max(sp_t[i]) > torch.max(sp_st[i])
        ts_most_conf[i] = out_t[i] if t_mask[i] else out_st[i]
    t_input = torch.sum(t_mask) / batch_size
    st_mask = torch.logical_not(t_mask)

    loss_teacher = kl_loss_fn(out_s[t_mask], out_t[t_mask], reduction=False)
    loss_student_teacher = kl_loss_fn(out_s[st_mask], out_st[st_mask], reduction=False) * gamma
    loss_teacher_sum = torch.sum(loss_teacher)
    loss_student_teacher_sum = torch.sum(loss_student_teacher)
    kl_total_loss = (torch.sum(loss_teacher) + torch.sum(loss_student_teacher)) / batch_size

    loss_combined = kl_loss_fn(out_s, ts_most_conf, reduction=False)
    loss_combined_sum = torch.sum(loss_combined)
    kl_loss = loss_combined_sum / batch_size

    st = 0
    t = 0
    for i in range(batch_size):
        if t_mask[i]:
            loss_mt = loss_teacher[t]
            t += 1
        else:
            loss_mt = loss_student_teacher[st]
            st += 1

        if not torch.all(torch.eq(loss_combined[i], loss_mt)):
            print(f'Difference in loss for image {i}: \n {loss_combined[i]} \n {loss_mt}')
    test


def adaptive_mt(device='cpu'):
    batch_size = 128
    kl_loss_fn = DistillKL(1)
    gamma = 1

    labels = torch.randint(0, 4, [batch_size], device=device)
    logit_s = sim_model_output(0.8, labels, 5).to(device)
    logit_st = sim_model_output(0.7, labels, 5).to(device)
    logit_t = sim_model_output(0.9, labels, 5).to(device)

    weighting_params = torch.randn(128, device=device)
    feat_s = torch.randn((batch_size, 128), device=device)
    feat_t = torch.randn((batch_size, 128), device=device)
    feat_st = torch.randn((batch_size, 128), device=device)

    sp_s = nn.functional.softmax(logit_s, dim=1)
    _, s_preds = torch.max(sp_s, 1)
    sp_st = nn.functional.softmax(logit_st, dim=1)
    _, st_preds = torch.max(sp_st, 1)

    teacher_weighting = torch.zeros((batch_size, 2), device=device)
    for i in range(batch_size):
        tmp = torch.mul(feat_st[i], feat_s[i])
        teacher_weighting[i][0] = torch.dot(weighting_params, torch.mul(feat_st[i], feat_s[i]))
        teacher_weighting[i][1] = torch.dot(weighting_params, torch.mul(feat_t[i], feat_s[i]))

    sp_tw = nn.functional.softmax(teacher_weighting, dim=1)
    _, t_mask = torch.max(sp_tw, 1)
    t_mask = t_mask.to(torch.bool)
    t_input = torch.sum(t_mask) / batch_size * 100

    st_mask = torch.logical_not(t_mask)
    loss_t = kl_loss_fn(logit_s[t_mask], logit_t[t_mask], reduction=False)
    loss_st = kl_loss_fn(logit_s[st_mask], logit_st[st_mask], reduction=False) * float(gamma)
    kl_loss = (torch.sum(loss_t) + torch.sum(loss_st)) / batch_size

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


def random_search(cfg, search_id=123):
    # runs = [(211, 132), (24, 160), (33, 261), (28, 232), (2, 171), (51, 267)]
    runs = [(88, 186), (77, 242), (95, 88), (157, 160)]
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
    print(f'Len Grid: {len(grid)}')
    params = grid[search_id]
    cfg.loss.N = float(params[1])
    cfg.loss.tau = float(params[2])
    cfg.student_id = int(params[0])

    return cfg


def random_search_test():
    cfg = OmegaConf.create({'optimizer': {'lr': 0},
                            'loss': {'tau': 0, 'alpha': 0, 'k': 0, 'kd_T': 0, 'N': 0, 'gamma': 0},
                            'student_id': 0, 'teacher_id': 0})
    param_combinations = []
    for i in range(48):
        #new_cfg = contdist_grid_search(cfg, search_id=i)
        new_cfg = get_teacher_student_id(cfg, i)
        print(f'{i}: {new_cfg}')
        param_combinations.append(
            [new_cfg.optimizer.lr, new_cfg.loss.tau, new_cfg.loss.N, new_cfg.loss.alpha, new_cfg.loss.k,
             new_cfg.loss.kd_T, new_cfg.loss.gamma, new_cfg.teacher_id, new_cfg.student_id])
    print(f'unique combinations: {len(np.unique(param_combinations, axis=0)) / 4}')
    print(f'evaluated runs: {len(np.unique(param_combinations, axis=0))}')


def sample_small_test_set(size=5, seed=12345):
    models = pd.read_csv('../files/contdist_model_list.csv')
    np.random.seed(seed)
    models = models.loc[(models['modelparams'] > 10) & (models['modelparams'] < 100)]
    models = models.sort_values(by=['modelparams'])
    architectures = np.unique(models['modeltype'].values)
    subsets = {}

    for arch in architectures:
        models_subset = models.loc[models['modeltype'] == arch]
        models_subset = models_subset.reset_index()
        n_models = len(models_subset.index)
        block_size = int(n_models / size)

        sample = []
        for b in range(size):
            tmp = models_subset.index[block_size * b:block_size * (b + 1)]
            sample.append(np.random.choice(tmp))

        subsets[arch] = models_subset.iloc[sample]
        print(f'{arch} students: {sample}')

    test


def get_teacher_student_id(cfg, experiment_id):
    students = [28, 5, 46, 261, 26, 171, 235, 92, 132, 42, 318, 9, 77, 258, 72, 299]
    teachers = [211, 234, 209, 10, 2, 80, 36, 182, 310, 77, 12, 239, 151, 145, 232, 101, 291, 1, 214, 124]

    s_t = []
    for s in students:
        for t in teachers:
            s_t.append([s, t])

    cfg.teacher_id = int(s_t[experiment_id][1])
    cfg.student_id = int(s_t[experiment_id][0])

    return cfg


def get_flips_per_class(preds_a, preds_b, true_y):
    pos_flips = np.zeros((len(true_y), 1000))
    neg_flips = np.zeros((len(true_y), 1000))
    for i in range(len(true_y)):
        pos_flips[i, true_y] = preds_a[i] != true_y[i] and preds_b[i] == true_y[i]
        neg_flips[i, true_y] = preds_a[i] == true_y[i] and preds_b[i] != preds_a[i]

    pos_flips = np.mean(pos_flips, axis=0)
    neg_flips = np.mean(neg_flips, axis=0)

    return pos_flips, neg_flips


def class_flips():
    logits = [np.random.rand(100, 1000), np.random.rand(100, 1000)]
    true_y = np.random.choice(range(1000), 100)
    preds = [np.argmax(l, axis=1) for l in logits]

    p, n = get_flips_per_class(preds[0], preds[1], true_y)
    test


def get_topk_class_sim(pos_class_flips, k=None, p=None):
    assert not(k is None and p is None), 'Please pass either k or p'
    sorted_classes = np.argsort(pos_class_flips)[::-1]

    if k is None:
        k = []
        for p_share in p:
            share, k_val = 0, 1
            while share < p_share:
                share = np.sum(pos_class_flips[sorted_classes[:k_val]]) / np.sum(pos_class_flips) * 100
                k_val += 1
            k.append(k_val)
    max_k = max(k)

    with open("../files/imagenet1000_clsidx_to_labels.txt") as f:
        idx2label = eval(f.read())

    class_names = [idx2label[c] for c in sorted_classes[:max_k]]

    import clip
    device = torch.device('cuda')
    model, _ = clip.load('ViT-B/32', device, jit=False)
    text_tokens = clip.tokenize(class_names).to(device)
    with torch.no_grad():
        text_features = torch.nn.functional.normalize(model.encode_text(text_tokens), dim=-1)  # Top-k x Dim
    sims = text_features @ text_features.T
    tmp = sims.cpu().numpy()
    avg_sim = []
    max_sim = []
    share_of_flips = []
    for top_k in k:
        i, j = [], []
        for l in range(top_k):
            i += [l]*(top_k-l-1)
            j += range(l+1, top_k)
        avg_sim.append(sims[i, j].mean().item())
        max_sim.append(sims[i, j].max().item())
        share_of_flips.append(np.sum(pos_class_flips[sorted_classes[:top_k]]) / np.sum(pos_class_flips) * 100)
    return k, avg_sim, max_sim, share_of_flips


def multi_teacher_dist(strat=''):
    imgs = torch.randn(128, 1)
    device = 'cpu'
    n_teachers = 4
    kl_loss_fn = DistillKL(1)

    labels = torch.randint(0, 4, [imgs.size(0)], device=device)
    out_s = sim_model_output(0.8, labels, 5).to(device)
    sp_st = sim_model_output(0.7, labels, 5).to(device)
    sp_t = [sim_model_output(0.9, labels, 5).to(device) for t in range(n_teachers)]

    t_mask = torch.zeros(imgs.size(0), device=device, dtype=torch.int)
    out_mt = torch.randn((imgs.size(0), 5), device=device)
    for i in range(imgs.size(0)):
        outputs = torch.zeros(n_teachers + 1, device=device)
        if '_u' in strat:
            outputs[0] = torch.max(sp_st[i])
            for t in range(n_teachers):
                outputs[t + 1] = torch.max(sp_t[t][i])
            argmax = torch.argmax(outputs)
            t_mask[i] = argmax - 1 if argmax > 0 else -1
        else:
            outputs[0] = sp_st[i][labels[i]]
            for t in range(n_teachers):
                outputs[t + 1] = sp_t[t][i][labels[i]]
            argmax = torch.argmax(outputs)
            t_mask[i] = argmax - 1

        out_mt[i] = sp_t[t_mask[i]][i] if t_mask[i] >= 0 else sp_st[i]
    kl_loss = kl_loss_fn(out_s, out_mt)
    t_input = (torch.sum(t_mask >= 0) / imgs.size(0)) * 100
    print(f'teacher input: {t_input}')


if __name__ == "__main__":
    multi_teacher_dist()
    test