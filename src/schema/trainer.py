"""Modeling for models training loop"""

import torch as _torch
from torch import nn as _nn
from torch.nn import functional as _F
from torch.utils.data import DataLoader as _DataLoader
from torch.optim import Optimizer as _Optimizer
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
from torchmetrics.classification import MulticlassF1Score as _MulticlassF1Score
from tqdm import tqdm as _tqdm


class Trainer:
    """Trainer class for automated training loop"""

    def __init__(
        self,
        model: _nn.Module,
        dataloader: _DataLoader,
        optimizer: _Optimizer,
        lr_scheduler: _LRScheduler,
        loss_fn,
        epochs: int,
        device=None,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs

    def train(self):
        dataloader_len = len(self.dataloader)
        print(f"training on device={self.device}")

        for epoch in range(self.epochs):
            batches_bar = _tqdm(self.dataloader, desc=f"epoch={epoch}/{self.epochs}")
            epoch_loss = 0
            for batch in batches_bar:
                self.optimizer.zero_grad()

                features, targets = batch
                features = _torch.as_tensor(features, device=self.device)
                targets = _torch.as_tensor(targets, device=self.device)

                logits = self.model(features)
                loss = self.loss_fn(logits, targets)
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()

                epoch_loss += loss.item()

                batches_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_avg_loss = epoch_loss / dataloader_len

            print(f"\navg_loss={epoch_avg_loss:.4f}")

    def save_model(self, path: str):
        _torch.save(self.model.state_dict(), path)
