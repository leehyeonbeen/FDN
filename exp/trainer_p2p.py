import argparse
import logging
import os
import sys
import warnings
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.amp import GradScaler

from data.dataset import *
from exp.helper import *
from exp.trainer import Trainer
from utils.loss import *
from utils.metrics import *
from utils.snippets import *

sys.path.append(os.getcwd())


# Suppress torch._dynamo warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

# Set logging levels to suppress dynamo messages
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)


class TrainerPoint2Point(Trainer):
    def __init__(
        self,
        model: nn.Module,
        configs: argparse.Namespace,
        dataset_cls=HydraulicDatasetAbsPos6D_NonForecasting,
        collate_fn_cls=CollatePoint2Point,
    ) -> None:
        super().__init__(
            model, configs, dataset_cls=dataset_cls, collate_fn_cls=collate_fn_cls
        )

    @func_timer
    def _setup_dataloader(
        self,
        dataset_cls=HydraulicDatasetAbsPos6D_NonForecasting,
        collate_fn_cls=CollatePoint2Point,
        index_stride=1,
        drop_last=True,
    ):
        super()._setup_dataloader(
            dataset_cls=dataset_cls,
            collate_fn_cls=collate_fn_cls,
            index_stride=index_stride,
            drop_last=drop_last,
        )

    @func_timer
    def _initialize_training(self):
        self.done_iters = 0
        self.done_epochs = 0
        self.tag_resumed = ""
        self.resumed_epoch = 0
        if torch.cuda.is_available():
            self.configs.device = "cuda"
        elif torch.mps.is_available():
            self.configs.device = "mps"
        # prepare model
        self.model = self.model.to(self.configs.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.grad_scaler = GradScaler(device=self.configs.device)

        # LRRT
        if self.enable_lrrt:
            self._find_lr()

        # AMP & Model Compiling
        self.amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if not self.disable_amp
            else nullcontext()
        )

        # Loss functions
        self.lossfn_mse = nn.MSELoss()
        # self.lossfn_gaussian_nll = GaussianNLLLoss()

        # LR scheduler
        if self.enable_lr_scheduling:
            max_lrs = [g["lr"] for g in self.optimizer.param_groups]
            if self.train_iters not in [0, None]:
                total_steps = self.train_iters
            else:
                total_steps = self.train_epochs * len(self.dataloader_train)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
            )

        print_line()
        for k, v in sorted(vars(self.configs).items()):
            print(f"PARAM {k} = {v}")
        print_line()

    def _train_step(self, batch_x, batch_y, batch_y_res, batch_y_trend, channel_mask):
        # AMP wrapping forward pass and loss computation
        with self.amp_context:
            # forward
            pred_y = self.model(batch_x)

            # restore FP32 accuracy
            pred_y = pred_y.to(torch.float32)
            batch_y = batch_y.to(torch.float32)
            # Deterministic model loss
            loss_trend = (
                self.lossfn_mse(pred_y, batch_y) * self.configs.loss_scale_trend
            )
            loss_res = torch.tensor(0.0)
            # logvar regularization
            loss_reg = torch.tensor(0.0)

            # Total loss
            loss = loss_trend + loss_res + loss_reg

        for p in self.model.parameters():
            p.grad = None

        # AMP backward
        if not self.disable_amp:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        # Non-AMP backward
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        if self.enable_lr_scheduling:
            self.scheduler.step()

        # DEBUG: check if non-updated parameters exist
        if self.done_iters == 100:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    assert not torch.all(p.grad == 0), (
                        f"Parameter {n} has zero gradient!"
                    )

        return loss_trend, loss_res, loss_reg


# class TrainerSeq2Point(TrainerPoint2Point):
#     def __init__(self, model: nn.Module, configs: argparse.Namespace) -> None:
#         super().__init__(model, configs)

#     @func_timer
#     def _setup_dataloader(self):
#         super()._setup_dataloader(collate_fn_cls=CollateSeq2Point)
