import argparse
import gc
import logging
import os
import sys
import time
import warnings
from contextlib import nullcontext
from copy import deepcopy

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch_lr_finder import LRFinder, TrainDataLoaderIter
from torchinfo import summary
from tqdm import tqdm

from data.dataset import *
from exp.helper import *
from exp.trainer import Trainer
from models.initialization import initialize_params
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


class TrainerGPR(Trainer):
    def __init__(self, model: nn.Module, configs: argparse.Namespace) -> None:
        super().__init__(
            model, configs, HydraulicDatasetAbsPos6D_NonForecasting, CollatePoint2Point
        )

    @func_timer
    def _setup_dataloader(
        self,
        dataset_cls=HydraulicDatasetAbsPos6D_NonForecasting,
        collate_fn_cls=CollatePoint2Point,
    ):
        super()._setup_dataloader(
            dataset_cls=dataset_cls,
            collate_fn_cls=collate_fn_cls,
            index_stride=1,
            drop_last=False,
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

        #############################################################################
        #############################################################################
        #############################################################################
        # initialize GPR model
        # https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html
        train_x = []
        train_y = []
        num_inducing = self.configs.num_inducing
        for i, (batch_x, batch_y, _, _, _) in enumerate(self.dataloader_train):
            train_x.append(batch_x)
            train_y.append(batch_y)
            print(f"Loading training batch: {i + 1}/{len(self.dataloader_train)}")
            if (i + 1) * self.dataloader_train.batch_size >= num_inducing:
                break
        self.train_x = torch.cat(train_x, dim=0)
        self.train_y = torch.cat(train_y, dim=0)
        self.model = self.model(self.configs, self.train_x[:num_inducing, :])
        # To GPU
        self.train_x = self.train_x.to(self.configs.device)
        self.train_y = self.train_y.to(self.configs.device)
        self.model = self.model.to(self.configs.device)
        # Loss
        self.lossfn_mll = gpytorch.mlls.VariationalELBO(
            self.model.likelihood, self.model, num_data=self.train_y.size(0)
        )
        #########################################################################
        #############################################################################
        #############################################################################

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )
        self.grad_scaler = GradScaler(device=self.configs.device)

        # LRRT
        if self.enable_lrrt:
            self._find_lr()

        # AMP & Model Compiling
        # self.amp_context = (
        #     torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        #     if not self.disable_amp
        #     else nullcontext()
        # )
        self.amp_context = nullcontext()  # GPR training is not compatible with AMP

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

    def fit(
        self,
        event_name=None,
    ):
        if hasattr(self.configs, "random_seed"):
            fix_random_seed(self.configs.random_seed)
        else:
            fix_random_seed(0)
        # torchinfo summary
        # if "Point2Point" in self.collate_fn.__class__.__name__:
        #     input_size = (self.batch_size, self.model.enc_in)
        # else:
        #     input_size = [
        #         (
        #             self.batch_size,
        #             self.configs.seq_len,
        #             self.model.enc_in,
        #         ),
        #         (
        #             self.batch_size,
        #             self.configs.label_len + self.configs.pred_len,
        #             self.configs.dec_in,
        #         ),
        #     ]
        # summary(
        #     self.model,
        #     input_size=input_size,
        #     col_names=(
        #         "num_params",
        #         "params_percent",
        #         "mult_adds",
        #         "trainable",
        #         "input_size",
        #         "output_size",
        #     ),
        #     col_width=15,
        #     depth=1,
        #     device=self.configs.device,
        # )
        self.event_name = event_namer(self.model, event_name)
        self._initialize_tb_logging()

        nan_flag = False
        lap0 = time.perf_counter()

        # training loop
        for epoch in range(self.train_epochs):
            self.model.train()
            with tqdm(
                total=len(self.dataloader_train),
                desc=f"Epoch {epoch + 1}/{self.train_epochs} (Training)",
                dynamic_ncols=True,
            ) as pbar:
                for it, batch in enumerate(self.dataloader_train):
                    batch_x, batch_y, batch_y_trend, batch_y_res, channel_mask = batch
                    batch_x = batch_x.to(self.configs.device)
                    batch_y = batch_y.to(self.configs.device)
                    batch_y_res = batch_y_res.to(self.configs.device)
                    batch_y_trend = batch_y_trend.to(self.configs.device)
                    channel_mask = channel_mask.to(self.configs.device)

                    lap1 = time.perf_counter()
                    # Forward pass
                    with self.amp_context:
                        pred_y = self.model(batch_x)
                        loss = -self.lossfn_mll(pred_y, batch_y)
                    # Zero gradients
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
                    if self.done_iters < 10:
                        for n, p in self.model.named_parameters():
                            if p.requires_grad:
                                assert not torch.all(p.grad == 0), (
                                    f"Parameter {n} has zero gradient!"
                                )

                    lap2 = time.perf_counter()
                    elapsed = lap2 - lap0
                    h, m, s = sec2hms(elapsed)
                    self.done_iters += 1

                    self.writer.add_scalar(
                        "training_loss/GPR_VariationalELBO",
                        loss.item(),
                        self.done_iters,
                    )
                    for i, g in enumerate(self.optimizer.param_groups):
                        self.writer.add_scalar(
                            f"training_loss/lr{i + 1:02d}", g["lr"], self.done_iters
                        )

                    pbar.update(1)
                    postfix_str = f"Loss=[{loss.item():.3e}], lr=["
                    for p in self.optimizer.param_groups:
                        postfix_str += f"{p['lr']:.2e}, "
                    postfix_str = postfix_str[:-2]  # remove last comma
                    postfix_str += (
                        f"], elapsed {h:02d}h {m:02d}m, {self.done_iters:,} its"
                    )
                    pbar.set_postfix_str(postfix_str)
                    if it == 0:
                        tqdm.write(f"Path: {self.event_name}")
                    nan_flag = loss.isnan()
                    if nan_flag:
                        raise ValueError(
                            f"NaN detected: loss={loss.item():.3e}, stopping training",
                        )

                        break
                    # break
            self.done_epochs += 1
            del (
                batch_x,
                batch_y,
                batch_y_trend,
                batch_y_res,
                channel_mask,
                pred_y,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if torch.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()

            # save every epoch
            if self.train_iters:
                pass
            else:
                self._save_checkpoint(
                    f"{self.event_name}/E{self.done_epochs}{self.tag_resumed}.pt",
                )

            if getattr(self, "flag_end_training", False):
                break

        self.writer.flush()
        self.writer.close()

        return self.model
