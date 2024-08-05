import os

import hydra

import wandb
import time
import torch
import logging

import pandas as pd
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import seed_everything
from distillation.data import get_ffcv_val_loader, get_ffcv_train_loader, get_cub_loader, get_caltech_loader, get_domainnet_loader, get_cars_loader
from distillation.models import init_timm_model, freeze_all_but_linear
from distillation.dist_utils import get_batch_size, AverageMeter, norm_batch, parse_cfg
from distillation.dist_utils import get_model, get_teacher_student_id, CosineAnnealingLRWarmup


class SupervisedTrainer:
    def __init__(self, cfg, model_name):
        seed_everything(cfg.seed)
        self.cfg = cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # initialize model
        self.model_name = model_name
        self.net, self.cfg_m = init_timm_model(model_name, self.device, pretrained='Finetune' in cfg.mode, num_classes=cfg.data.num_classes)

        # freeze encoder layers
        if cfg.freeze:
            freeze_all_but_linear(self.net)
            # check if model backbone is frozen
            for name, param in self.net.named_parameters():
                if 'fc' not in name and 'classifier' not in name and 'head' not in name:
                    assert not param.requires_grad
        logging.info(f'Memory usage: {torch.cuda.memory_allocated(self.device) / (1024 ** 2)}')
        logging.info(f'Model: {model_name}, config: {self.cfg_m}')

        # initialize the validation and train dataloaders
        if cfg.data.dataset == 'ImageNet':
            self.val_loader = get_ffcv_val_loader(self.cfg_m, self.device, cfg, batch_size=cfg.optimizer.batch_size)
            self.train_loader = get_ffcv_train_loader(self.cfg_m, self.device, cfg)
        elif cfg.data.dataset == 'CUB':
            self.val_loader = get_cub_loader(self.cfg, self.cfg_m, is_train=False)
            self.train_loader = get_cub_loader(self.cfg, self.cfg_m, is_train=True)
        elif cfg.data.dataset == 'caltech':
            self.val_loader = get_caltech_loader(self.cfg, self.cfg_m, is_train=False)
            self.train_loader = get_caltech_loader(self.cfg, self.cfg_m, is_train=True)
        elif cfg.data.dataset == 'infograph':
            self.val_loader = get_domainnet_loader(self.cfg, self.cfg_m, is_train=False)
            self.train_loader = get_domainnet_loader(self.cfg, self.cfg_m, is_train=True)
        elif cfg.data.dataset == 'cars':
            self.val_loader = get_cars_loader(self.cfg, self.cfg_m, is_train=False)
            self.train_loader = get_cars_loader(self.cfg, self.cfg_m, is_train=True)
        else:
            raise NotImplementedError(f'{cfg.data.dataset} is not implemented yet.')

        # initialize loss function
        self.loss_function = nn.CrossEntropyLoss()

        # initialize optimizer and lr scheduler
        scale_lr = cfg.optimizer.lr * cfg.optimizer.batch_size / 256
        if cfg.optimizer.name == 'adamw':
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=scale_lr,
                                               weight_decay=cfg.optimizer.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=scale_lr, momentum=cfg.optimizer.momentum,
                                             weight_decay=cfg.optimizer.weight_decay)
        self.iter_per_epoch = len(self.train_loader)
        if cfg.scheduler.name == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=cfg.scheduler.warmup,
                                                                                  T_mult=1, eta_min=cfg.scheduler.eta_min)
        elif cfg.scheduler.name == 'CosineAnnealingLRwWarmup':
            scale_min_lr = cfg.scheduler.eta_min * cfg.optimizer.batch_size / 256
            self.scheduler = CosineAnnealingLRWarmup(self.optimizer, cfg.max_epochs * self.iter_per_epoch,
                                                     warmup_iters=(cfg.scheduler.warmup / 100) * cfg.max_epochs * self.iter_per_epoch,
                                                     min_lr=scale_min_lr)
        else:
            self.scheduler = None

        # initialize grad scales for half precision
        if cfg.data.precision == 'half':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.checkpoint_path = os.path.join(cfg.checkpoint.dir, cfg.data.dataset, model_name + '_lin_ft' if cfg.freeze else model_name)

        self.n_iter = 0

    def load_from_checkpoint(self):
        """Load distillation run from checkpoint

        Returns: tuple (epoch, loss, wandb_id)
            - epoch: Latest epoch of the checkpoint
            - loss: Latest loss of the checkpoint
            - wandb_id: ID of the wandb run

        """
        checkpoint = torch.load(self.checkpoint_path + f'/{self.model_name}.pt')

        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        self.n_iter = checkpoint['n_iter']
        loss = checkpoint['loss']
        wandb_id = checkpoint['wandb_id']
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return epoch + 1, loss, wandb_id

    def load_pretrain_weights(self):
        checkpoint = torch.load(self.checkpoint_path + f'/{self.model_name}.pt')

        self.net.load_state_dict(checkpoint['model_state_dict'])

    def save_to_checkpoint(self, epoch, loss, wandb_id):
        """Save distillation run to checkpoint

        :param epoch: Latest epoch
        :param loss: Latest loss
        :param wandb_id: ID of the wandb run

        Returns:

        """

        model_state_dict = self.net.state_dict()

        scheduler_dict = self.scheduler.state_dict() if self.scheduler is not None else None
        torch.save({'epoch': epoch,
                    'n_iter': self.n_iter,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler_dict,
                    'loss': loss,
                    'wandb_id': wandb_id},
                   self.checkpoint_path + f'/{self.model_name}.pt')

    def train_one_epoch(self, epoch):
        """Train one epoch

        :param epoch: Index of current epoch

        Returns: dict {train_loss, train_acc}

        """
        start = time.time()
        self.net.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        for batch_index, batch_inputs in enumerate(self.train_loader):
            images, labels, idxs = batch_inputs
            batch_time = time.time()

            labels = labels.cuda()
            images = images.cuda()

            self.optimizer.zero_grad()
            if self.cfg.data.precision == 'half':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.net(images)
                    loss = self.loss_function(outputs, labels)
            else:
                outputs = self.net(images)
                loss = self.loss_function(outputs, labels)

            # check if loss is nan
            if torch.isnan(loss):
                logging.info('Loss is NaN')
                logging.info(f'Epoch: {epoch}, Iteration: {batch_index}')
                logging.info(f'Model output: {outputs}')
                raise ValueError('Loss is NaN')

            if self.cfg.data.precision == 'half':
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            _, preds = outputs.max(1)
            correct = preds.eq(labels).sum()
            self.n_iter += len(labels)

            log = {'train_loss': loss.item(), 'lr': self.optimizer.param_groups[0]['lr'],
                   'time': (time.time() - batch_time) / len(labels)}
            wandb.log(log, step=self.n_iter)

            if self.scheduler is not None:
                self.scheduler.step()

            train_loss.update(loss.item(), images.size(0))
            train_acc.update(correct.item() / images.size(0), images.size(0))

        finish = time.time()
        logging.info('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
        wandb.log({'train_acc': train_acc.avg * 100}, step=self.n_iter)
        return {'train_loss': train_loss.avg, 'train_acc': train_acc.avg * 100}

    @torch.no_grad()
    def eval_training(self):
        """Evaluate training

        Returns: dict {test_loss, test_acc}

        """

        self.net.eval()

        test_loss = 0.0
        correct = 0.0
        n_imgs = 0.0
        predictions = []

        for batch_inputs in self.val_loader:
            images, labels, idxs = batch_inputs
            images = images.cuda()
            labels = labels.cuda()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.net(images)
                loss = self.loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
            n_imgs += len(labels)
            predictions += preds.cpu().tolist()
        log = {'test_loss': test_loss / n_imgs,
               'test_acc': correct.float() / n_imgs * 100}
        wandb.log(log, step=self.n_iter)
        return log


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
    models_list = pd.read_csv('files/timm_model_list.csv')

    # get teacher and student model names and parameters from config
    model_name, model_type, model_params = get_model(cfg.model_id, models_list)

    # get suitable batch size
    if cfg.optimizer.batch_size == 'auto':
        batch_size = get_batch_size(model_name, model_name, device, cfg.loss.name)
        cfg.optimizer.batch_size = batch_size
        config['batch_size'] = batch_size
    cfg = parse_cfg(cfg)

    # initialize the supervised trainer
    trainer = SupervisedTrainer(cfg, model_name)

    st_cfg = {'model_name': model_name, 'model_type': model_type, 'model_params': model_params}
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
        wandb.run.name = f'pretrain-{model_name}'
        logging.info(f'Loaded from checkpoint: {trainer.checkpoint_path}')
    except (FileNotFoundError, RuntimeError):
        # create checkpoint folder
        os.makedirs(trainer.checkpoint_path, exist_ok=True)
        # initialize wandb logger
        wandb_id = wandb.util.generate_id()
        wandb.init(id=wandb_id, project=cfg.wandb.project, config=config, tags=tags)
        wandb.run.name = f'pretrain-{model_name}'
        e_start = 0

    for e in range(e_start, cfg.max_epochs):
        logging.info(f'Train Epoch {e}')
        train_stats = trainer.train_one_epoch(e)
        if trainer.val_loader is not None:
            test_stats = trainer.eval_training(e)
            logging.info(f'Train Loss: {train_stats["train_loss"]:.4f} | Test Loss: {test_stats["test_loss"]:.4f}')
            logging.info(f'Train Acc: {train_stats["train_acc"]:.4f} | Test Acc: {test_stats["test_acc"]:.4f}')
        trainer.save_to_checkpoint(e, train_stats['train_loss'], wandb_id)


if __name__ == '__main__':
    main()
