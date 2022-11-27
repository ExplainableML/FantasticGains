# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from solo.losses.simclr import simclr_loss_func
from solo.methods.base import BaseMethod


class SimCLR(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        
        self.count = 0

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SimCLR, SimCLR).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """
        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Dict[int, Sequence[Any]], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Dict[int,Sequence[Any]]): dict comprising a batch of data (or collection thereof if we have multiple dataloaders) in the format of [img_indexes, [X], Y], 
                where [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """
        # if self.count % 100 == 0:
        #     import numpy as np
        #     import solo
        #     mean, std = solo.utils.constants.MEANS_N_STD.get('imagenet100')       
        #     samples_1 = batch[0][1][0].detach().cpu().numpy().astype(np.float32)
        #     samples_2 = batch[0][1][1].detach().cpu().numpy().astype(np.float32)
        #     # mean, std = solo.utils.constants.FFCV_MEANS_N_STD.get('imagenet100')
        #     # samples_1 = batch[0][0].detach().cpu().numpy().astype(np.float32)
        #     # samples_2 = batch[1][0].detach().cpu().numpy().astype(np.float32)
        #     n_samples = 10
        #     rand_idcs = np.random.choice(len(samples_1), n_samples, replace=False)
        #     samples_1 = np.clip(((samples_1 * np.array(std).reshape(1, 3, 1, 1)) + np.array(mean).reshape(1, 3, 1, 1)) * 255, 0, 255).astype(np.uint8)
        #     samples_2 = np.clip(((samples_2 * np.array(std).reshape(1, 3, 1, 1)) + np.array(mean).reshape(1, 3, 1, 1)) * 255, 0, 255).astype(np.uint8)
        #     # samples_1 = np.clip(((samples_1 * np.array(std).reshape(1, 3, 1, 1)) + np.array(mean).reshape(1, 3, 1, 1)) * 1, 0, 255).astype(np.uint8)
        #     # samples_2 = np.clip(((samples_2 * np.array(std).reshape(1, 3, 1, 1)) + np.array(mean).reshape(1, 3, 1, 1)) * 1, 0, 255).astype(np.uint8)
        #     import matplotlib.pyplot as plt
        #     f, axes = plt.subplots(2, len(rand_idcs))
        #     for i in range(len(rand_idcs)):
        #         axes[0, i].imshow(samples_1[i].transpose(1, 2, 0))
        #         axes[1, i].imshow(samples_2[i].transpose(1, 2, 0))
        #     f.set_size_inches(4 * len(rand_idcs), 4 * 2)
        #     f.tight_layout()
        #     f.savefig(f'sample_viz_{self.count}_base.png')
        #     plt.close() 
        self.count += 1
        batch = super().prepare_batch(batch)
        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        # out = super().training_step([batch[0][-1].clone(), [batch[0][0].clone(), batch[1][0].clone()], batch[0][1].clone()], batch_idx)
        class_loss = out["loss"]
        z = torch.cat(out["z"])
        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        nce_loss = simclr_loss_func(
            z,
            indexes=indexes,
            temperature=self.temperature,
        )
        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss
