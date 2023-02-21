import torch
import logging

import torch.nn.functional as F
import numpy as np
from torchvision import transforms

from sklearn.metrics import accuracy_score
from omegaconf import DictConfig, OmegaConf

from .models import init_timm_model


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


def get_batch_size(teacher_name, student_name, device, loss, max_batch_size=None, num_iterations=5) -> int:
    student, _ = init_timm_model(student_name, device)
    teacher, _ = init_timm_model(teacher_name, device)
    student.train()
    teacher.eval()
    if 'mt' in loss:
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
                        if 'mt' in loss:
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
    if 'cd' in loss or 'crd' in loss:
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
    runs = [(33, 261), (28, 232), (51, 267)]
    param_grid = {
        'lr': [1e-2, 1e-3],
        'alpha': [0, 1],
        'k': [10, 100, 1000],
        'kd_T': [0.1, 1, 10]
    }

    f = 1

    np.random.seed(int(search_id / len(runs)) * f)
    cfg.optimizer.lr = float(np.random.choice(param_grid['lr']))
    cfg.loss.alpha = round(float(np.random.uniform(param_grid['alpha'][0], param_grid['alpha'][1])), 2)
    cfg.loss.k = int(np.random.choice(param_grid['k']))
    cfg.loss.kd_T = float(np.random.choice(param_grid['kd_T']))
    cfg.teacher_id = runs[search_id % len(runs)][0]
    cfg.student_id = runs[search_id % len(runs)][1]

    return cfg


def grid_search(cfg, id=123):
    runs = [397, 235, 63, 197, 105, 92]
    param_grid = {
        'lr': [1e-2, 1e-3, 1e-4, 1e-5],
        'tau': [0.9, 0.99, 0.999],
        'N': [2, 10, 50],
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


def get_model(id, models_list):
    name = models_list['modelname'][id]
    type = models_list['modeltype'][id]
    params = models_list['modelparams'][id]
    return name, type, params
