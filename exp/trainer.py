import argparse
import gc
import logging
import os
import sys
import time
import warnings
import weakref
from contextlib import nullcontext
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_lr_finder import LRFinder, TrainDataLoaderIter
from torchinfo import summary
from tqdm import tqdm

from data.dataset import *
from exp.helper import *
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


def _is_probabilistic_parameter(name: str) -> bool:
    return (
        "linear_mu" in name
        or "linear_logvar" in name
        or "residual_distribution" in name
    )


class Trainer:
    def __init__(
        self,
        model: type[nn.Module],
        configs: argparse.Namespace,
        dataset_cls=HydraulicDatasetRelPos7D,
        collate_fn_cls=CollateDecompose,
    ) -> None:
        self.configs = configs
        self.model = model
        self.train_iters = configs.train_iters
        self.train_epochs = configs.train_epochs
        self.batch_size = configs.batch_size
        self.lr = configs.lr
        self.enable_lrrt = configs.enable_lrrt
        self.enable_lr_scheduling = configs.enable_lr_scheduling
        self.disable_amp = configs.disable_amp
        self.disable_compile = configs.disable_compile
        self.cutoff_freq = configs.cutoff_freq
        self._setup_dataloader(dataset_cls=dataset_cls, collate_fn_cls=collate_fn_cls)
        self._initialize_training()

    @func_timer
    def _setup_dataloader(
        self,
        dataset_cls,
        collate_fn_cls,
        index_stride=1,
        drop_last=True,
    ):
        seq_len = self.configs.seq_len
        pred_len = self.configs.pred_len

        if collate_fn_cls is not None:
            self.collate_fn = collate_fn_cls(trend_cutoff_freq=self.cutoff_freq)
        else:
            self.collate_fn = None

        self.dataset_train = dataset_cls(
            seq_len,
            pred_len,
            "train",
            sampling_freq=self.configs.sampling_freq,
        )
        self.dataset_train.scale()

        generator = torch.Generator()
        generator.manual_seed(self.configs.random_seed)
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=min(4, os.cpu_count() // 3),
            persistent_workers=False,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )

        if hasattr(self.model, "enc_in"):
            assert self.dataset_train.n_inputs == self.model.enc_in, (
                f"Input dim mismatch: dataset={self.dataset_train.n_inputs}, "
                f"model.enc_in={self.model.enc_in}"
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

        # optimizer
        # Single Optimizer Setting
        params_probheads = [
            p
            for n, p in self.model.named_parameters()
            if _is_probabilistic_parameter(n)
        ]
        params_others = [
            p
            for n, p in self.model.named_parameters()
            if not _is_probabilistic_parameter(n)
        ]
        optimizer_targets = [
            {"params": params_others, "lr": self.lr},
            {
                "params": params_probheads,
                "lr": self.lr * self.configs.lr_scale_probheads,
            },
        ]
        self.optimizer = torch.optim.Adam(optimizer_targets)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
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
        # if not self.disable_compile:
        #   self.model = torch.compile(self.model)

        # Loss functions
        self.lossfn_mse = nn.MSELoss()
        self.lossfn_gaussian_nll = GaussianNLLLoss()

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

    def _gaussian_nll(self, target, mu, logvar):
        model = get_model(self.model)
        if hasattr(model, "gaussian_nll"):
            return model.gaussian_nll(target, mu, logvar)
        return self.lossfn_gaussian_nll(target, mu, logvar)

    def _initialize_tb_logging(self):
        # tensorboard writer
        self.filename_suffix = uuid.uuid4().hex[:8]
        self.writer = SummaryWriter(
            self.event_name, filename_suffix=self.filename_suffix
        )
        self.event_file_name = self.writer._get_file_writer().event_writer._file_name
        self.event_file_name = self.event_file_name.replace(f"{self.event_name}/", "")

    def _train_step(self, batch_x, batch_y, batch_y_res, batch_y_trend, channel_mask):
        batch_size, input_length, input_size = batch_x.size()
        batch_size, output_length, output_size = batch_y.size()

        # AMP wrapping forward pass and loss computation
        with self.amp_context:
            # forward
            pred_trend, pred_res, pred_mu, pred_logvar = self.model(
                batch_x, channel_mask
            )

            # restore FP32 accuracy
            pred_trend = pred_trend.to(torch.float32)
            pred_mu = pred_mu.to(torch.float32)
            pred_logvar = pred_logvar.to(torch.float32)
            batch_y_trend = batch_y_trend.to(torch.float32)
            batch_y_res = batch_y_res.to(torch.float32)

            if self.configs.disable_probhead:  # full-deterministic (ablation)
                loss_trend = self.lossfn_mse(pred_trend, batch_y)
                loss_res = torch.tensor(0.0)
                loss_reg = torch.tensor(0.0)
                loss = loss_trend
            elif self.configs.disable_dethead:  # full-probabilistic (ablation)
                loss_trend = torch.tensor(0.0)
                loss_res = (
                    self._gaussian_nll(batch_y, pred_mu, pred_logvar)
                    * self.configs.loss_scale_res
                )
                loss_reg = torch.tensor(0.0)
                loss = loss_res
            else:  # default model
                loss_trend = (
                    self.lossfn_mse(pred_trend, batch_y_trend)
                    * self.configs.loss_scale_trend
                )
                loss_res = (
                    self._gaussian_nll(batch_y_res, pred_mu, pred_logvar)
                    * self.configs.loss_scale_res
                )
                # logvar regularization
                loss_reg = torch.mean(
                    torch.square(F.relu(pred_logvar))
                    * self.configs.loss_scale_logvar_reg
                )  # penalize positive parts only

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

        # DEBUG: check if non-updated parameters exist after some iterations
        if self.done_iters == 100:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    assert not torch.all(p.grad == 0), (
                        f"Parameter {n} has zero gradient!"
                    )

        return loss_trend, loss_res, loss_reg

    def fit(
        self,
        event_name=None,
    ):
        if hasattr(self.configs, "random_seed"):
            fix_random_seed(self.configs.random_seed)
        else:
            fix_random_seed(0)
        # torchinfo summary
        if "Point2Point" in self.collate_fn.__class__.__name__:
            input_size = (self.batch_size, self.model.enc_in)
        else:
            input_size = (self.batch_size, self.configs.seq_len, self.model.enc_in)
        summary(
            self.model,
            input_size=input_size,
            col_names=(
                "num_params",
                "params_percent",
                "mult_adds",
                "trainable",
                "input_size",
                "output_size",
            ),
            col_width=15,
            depth=1,
            device=self.configs.device,
        )
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
                    loss_trend, loss_res, loss_reg = self._train_step(
                        batch_x, batch_y, batch_y_res, batch_y_trend, channel_mask
                    )
                    lap2 = time.perf_counter()
                    elapsed = lap2 - lap0
                    h, m, s = sec2hms(elapsed)
                    self.done_iters += 1

                    self.writer.add_scalar(
                        "training_loss/trend", loss_trend.item(), self.done_iters
                    )
                    self.writer.add_scalar(
                        "training_loss/res", loss_res.item(), self.done_iters
                    )
                    self.writer.add_scalar(
                        "training_loss/reg", loss_reg.item(), self.done_iters
                    )
                    self.writer.add_scalar(
                        "training_loss/total",
                        loss_trend.item() + loss_res.item() + loss_reg.item(),
                        self.done_iters,
                    )
                    for i, g in enumerate(self.optimizer.param_groups):
                        self.writer.add_scalar(
                            f"training_loss/lr{i + 1:02d}", g["lr"], self.done_iters
                        )

                    pbar.update(1)
                    postfix_str = (
                        f"Loss=[{loss_trend:.1e}, {loss_res:.1e}, {loss_reg:.1e}], lr=["
                    )
                    for p in self.optimizer.param_groups:
                        postfix_str += f"{p['lr']:.2e}, "
                    postfix_str = postfix_str[:-2]  # remove last comma
                    postfix_str += (
                        f"], elapsed {h:02d}h {m:02d}m, {self.done_iters:,} its"
                    )
                    pbar.set_postfix_str(postfix_str)
                    if it == 0:
                        tqdm.write(f"Path: {self.event_name}")
                    nan_flag = loss_trend.isnan() or loss_res.isnan()
                    if nan_flag:
                        raise ValueError(
                            f"NaN detected: loss_trend={loss_trend:.3e}, loss_res={loss_res:.3e}, stopping training",
                        )

                    # save every 10k iterations
                    if self.train_iters:
                        if self.done_iters % 10000 == 0:
                            self._save_checkpoint(
                                f"{self.event_name}/IT{self.done_iters // 1000:03d}K{self.tag_resumed}.pt",
                            )
                        if self.done_iters >= self.train_iters:
                            self.flag_end_training = True
                            break
                    # break
            self.done_epochs += 1
            del (
                batch_x,
                batch_y,
                batch_y_trend,
                batch_y_res,
                channel_mask,
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

    def _copy_state_dict(self, state_dict):
        return {
            k: v.detach().cpu() if torch.is_tensor(v) else v
            for k, v in state_dict.items()
        }

    def _save_checkpoint(self, fname: str):
        model_instance = get_model(self.model)
        torch.save(
            {
                "state_dict": self._copy_state_dict(model_instance.state_dict()),
                "configs": deepcopy(self.configs),
                "scale_params": deepcopy(self.dataset_train.scale_params),
                "optimizer": self._copy_state_dict(self.optimizer.state_dict()),
            },
            fname,
        )

    def resume_from_checkpoint(self, ckpt_path: str):
        self.ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        self.model.load_state_dict(self.ckpt["state_dict"])
        self.optimizer.load_state_dict(self.ckpt["optimizer"])
        self.tag_resumed = (
            f"_from_{os.path.basename(ckpt_path).replace('.pt', '')}"  # E7_from_E1.pt
        )
        self.done_epochs = int(
            os.path.basename(ckpt_path).split(".")[0].split("_")[0][1:]
        )
        self.done_iters = self.done_epochs * len(self.dataloader_train)
        self.train_epochs -= self.done_epochs
        for _ in range(self.done_iters):
            if self.enable_lr_scheduling:
                self.scheduler.step()
        print(f"Resumed training from checkpoint: {ckpt_path}")
        event_name = os.path.dirname(ckpt_path).split("exp/runs/")[-1]
        self.fit(event_name)

    def _find_lr(self):
        self.lr_finder = LRFinder(
            model=self.model,
            optimizer=self.optimizer,
            criterion=LRFinderLoss(self.model),
            device="cuda",
            amp_backend="torch",
            amp_config={"device_type": "cuda", "dtype": torch.bfloat16},
            grad_scaler=self.grad_scaler,
        )
        self.lr_finder.range_test(
            train_loader=TrainIter(self.dataloader_train),
            start_lr=1e-5,
            end_lr=1e-3,
            num_iter=1000,
            step_mode="exp",
        )  # finds the first param_group's lr only
        ax, self.lr = self.lr_finder.plot()
        self.configs.lr = self.lr
        self.lr_finder.reset()  # restores the model and optimizer states saved before the range test
        self.optimizer.param_groups[0]["lr"] = self.lr


class Pretrainer(Trainer):
    def __init__(
        self,
        model: type[nn.Module],
        configs: argparse.Namespace,
        dataset_cls=RH20TDatasetRelPos7D,
        collate_fn_cls=CollateDecompose,
        train_iters: int = 100000,
    ) -> None:
        self.configs = configs
        self.model = model
        self.configs.train_iters = train_iters
        self.train_iters = train_iters
        if self.train_iters:
            self.train_epochs = 1000  # large number to ignore epoch-based stopping
        else:
            self.train_epochs = configs.train_epochs
        self.batch_size = configs.batch_size
        self.lr = configs.lr
        self.enable_lrrt = configs.enable_lrrt
        self.enable_lr_scheduling = configs.enable_lr_scheduling
        self.disable_amp = configs.disable_amp
        self.disable_compile = configs.disable_compile
        self.cutoff_freq = configs.cutoff_freq
        self._setup_dataloader(dataset_cls=dataset_cls, collate_fn_cls=collate_fn_cls)
        self._initialize_training()

    @func_timer
    def _setup_dataloader(self, dataset_cls, collate_fn_cls):
        seq_len = self.configs.seq_len
        pred_len = self.configs.pred_len

        self.collate_fn = collate_fn_cls(trend_cutoff_freq=self.cutoff_freq)
        self.dataset_train = dataset_cls(
            seq_len,
            pred_len,
            "train",
            train_ratio=self.configs.train_ratio,
            index_stride=10,  # 10 steps interval among 100 Hz samples
        )
        self.dataset_train.scale()

        # setup dataloaders
        generator = torch.Generator()
        generator.manual_seed(self.configs.random_seed)
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=min(4, os.cpu_count() // 3),
            persistent_workers=False,
            drop_last=True,
            collate_fn=self.collate_fn,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )

        if hasattr(self.model, "enc_in"):
            assert self.dataset_train.n_inputs == self.model.enc_in, (
                f"Input dim mismatch: dataset={self.dataset_train.n_inputs}, "
                f"model.enc_in={self.model.enc_in}"
            )


class FineTuner(Trainer):
    def __init__(
        self,
        model: type[nn.Module],
        configs: argparse.Namespace,
        dataset_cls=HydraulicDatasetRelPos7D,
        collate_fn_cls=CollateDecompose,
        ckpt_path: str = None,
        mode: str = "linprobe",  #'linprobe' or 'finetune'
    ) -> None:
        self.configs = configs
        self.model = model
        self.train_iters = configs.train_iters
        self.train_epochs = configs.train_epochs
        self.batch_size = configs.batch_size
        self.lr = configs.lr
        self.enable_lrrt = configs.enable_lrrt
        self.enable_lr_scheduling = configs.enable_lr_scheduling
        self.disable_amp = configs.disable_amp
        self.disable_compile = configs.disable_compile
        self.pretrain_ckpt_path = ckpt_path
        self.cutoff_freq = configs.cutoff_freq
        self.mode = mode

        self._setup_dataloader(
            dataset_cls=dataset_cls,
            collate_fn_cls=collate_fn_cls,
        )
        self._initialize_training()
        self._initialize_fine_tuning()
        self._update_scale_params()

    def _initialize_fine_tuning(self):
        if self.pretrain_ckpt_path is not None:
            # Load pretrained encoders
            self.ckpt = torch.load(
                self.pretrain_ckpt_path, weights_only=False, map_location="cpu"
            )
            pretrain_state_dict = self.ckpt["state_dict"]
            self.pretrain_configs = self.ckpt["configs"]
            self.pretrain_scale_params = self.ckpt["scale_params"]
            self.model.load_state_dict(pretrain_state_dict)

            print("All parameters loaded.")

            if self.mode == "finetune":
                for p in self.model.parameters():
                    p.requires_grad = True
                print("All parameters unfreezed for fine-tuning")
            elif self.mode == "linprobe":
                initialize_params(self.model.freq_enhance)
                # initialize_params(self.model.encoder_jtorque) # initialized or pretrained. no need to control.
                initialize_params(self.model.channel_mixer)
                initialize_params(self.model.decoder)
                print("Initialized channel mixer and decoders")
                for n, p in self.model.named_parameters():
                    n = n.lower()
                    if ("jointpos" in n) or ("jointvel" in n) or ("jointacc" in n) or ("shared" in n):
                        p.requires_grad = False
                    elif (
                        not self.pretrain_configs.disable_jtorque and "jtorque" in n
                    ):  # if pretrained jtorque, freeze
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                print("Freezed JointPosVelAccTorque encoders")
        else:
            print(
                "No checkpoint path provided, training from the given model at __init__."
            )

        # Smaller LR for pretrained parameters
        params_jointpos = [
            p for n, p in self.model.named_parameters() if "encoder_jointpos" in n
        ]
        params_jointvel = [
            p for n, p in self.model.named_parameters() if "encoder_jointvel" in n
        ]
        params_jointacc = [
            p for n, p in self.model.named_parameters() if "encoder_jointacc" in n
        ]
        params_jtorque = [
            p for n, p in self.model.named_parameters() if "encoder_jtorque" in n
        ]
        params_probheads = [
            p
            for n, p in self.model.named_parameters()
            if _is_probabilistic_parameter(n)
        ]
        params_others = [
            p
            for n, p in self.model.named_parameters()
            if ("encoder_jointpos" not in n)
            and ("encoder_jointvel" not in n)
            and ("encoder_jointacc" not in n)
            and ("encoder_jtorque" not in n)
            and not _is_probabilistic_parameter(n)
        ]
        optimizer_targets = [
            {"params": params_others, "lr": self.lr},  # non-pretrained
            {
                "params": params_probheads,  # non-pretrained probabilistic heads
                "lr": self.lr * self.configs.lr_scale_probheads,
            },
        ]
        if not self.configs.freeze_jointq:  # add pretrained params if not frozen
            optimizer_targets.append(
                {
                    "params": params_jointpos + params_jointvel + params_jointacc,
                    "lr": self.lr * self.configs.lr_scale_pretrained,
                }
            )
        if not self.configs.freeze_jtorque:  # add pretrained params if not frozen
            optimizer_targets.append(
                {
                    "params": params_jtorque,
                    "lr": self.lr,  # * self.configs.lr_scale_pretrained,
                }
            )

        # Refind LR with pretrained params
        if self.enable_lrrt:
            self._find_lr()

        # Allocate low LR to pretrained params
        self.optimizer = torch.optim.Adam(optimizer_targets)

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

    def _update_scale_params(self):
        edited_scale_params = deepcopy(self.dataset_train.scale_params)

        # 7J + 7J + 7J + 7J + 7J0 = 35
        edited_scale_params["mean_i"][:21] = self.pretrain_scale_params["mean_i"][:21]
        edited_scale_params["std_i"][:21] = self.pretrain_scale_params["std_i"][:21]
        edited_scale_params["mean_i"][28:] = self.pretrain_scale_params["mean_i"][28:]
        edited_scale_params["std_i"][28:] = self.pretrain_scale_params["std_i"][28:]

        # save as attributes
        self.dataset_train.scale_params = edited_scale_params
        print("Updated scaling parameters for JointPosVelAcc channels.")


class LRFinderLoss(nn.Module):
    def __init__(self, model=None):
        super(LRFinderLoss, self).__init__()
        self.lossfn_mse = nn.MSELoss()
        self.lossfn_gaussian_nll = GaussianNLLLoss()
        self._model_ref = weakref.ref(get_model(model)) if model is not None else None

    def forward(self, input, target):
        pred_trend, pred_res, pred_mu, pred_logvar = input
        bo, batch_y_trend, batch_y_res = target

        loss_trend = self.lossfn_mse(pred_trend, batch_y_trend)
        model = self._model_ref() if self._model_ref is not None else None
        if model is not None and hasattr(model, "gaussian_nll"):
            loss_res = model.gaussian_nll(batch_y_res, pred_mu, pred_logvar)
        else:
            loss_res = self.lossfn_gaussian_nll(
                batch_y_res, pred_mu, pred_logvar
            )

        return loss_trend + loss_res


class TrainIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch):
        x, y, y_trend, y_res, ch_mask = batch
        return x, (y, y_trend, y_res)

