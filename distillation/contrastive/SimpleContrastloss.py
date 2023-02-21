import logging
import torch
import sys

import torch.nn.functional as F


class SimpleContrastLoss(torch.nn.Module):
    def __init__(self):
        super(SimpleContrastLoss, self).__init__()

    def forward(self, f_s, f_t):
        f_s = F.normalize(f_s, dim=1)
        f_t = F.normalize(f_t, dim=1)


        f_s_norm = f_s / f_s.norm(dim=1)[:, None]
        f_t_norm = f_t / f_t.norm(dim=1)[:, None]
        sim_s = torch.mm(f_s_norm, f_s_norm.transpose(0, 1))
        sim_t = torch.mm(f_t_norm, f_t_norm.transpose(0, 1))

        sim_s = F.log_softmax(sim_s, dim=1)
        sim_t = F.log_softmax(sim_t, dim=1)
        div = F.kl_div(sim_s, sim_t, log_target=True, reduction='sum')
        return div
