import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os, sys
import uuid

sys.path.append(os.getcwd())
import numpy as np
import datetime
from typing import Optional
import argparse
import platform
from utils.metrics import *

from utils.snippets import *
from utils.metrics import *
from utils.loss import GaussianNLLLoss
from layers.Filter import FreqPassFilter


def parse_exp_configs():
    parser = argparse.ArgumentParser(
        description="Frequency-aware Decomposition Network (FDN) for sensorless wrench estimation."
    )
    ##############################################################################################################################
    parser.add_argument(
        "-n",
        "--exp_name",
        default="test",
        type=str,
        help="experiment name, logged as exp/runs/<exp_name>",
    )
    parser.add_argument(
        "-t",
        "--exp_tag",
        default="",
        type=str,
        help="experiment tag, logged as exp/runs/<exp_name>/<RUN>_<exp_tag>",
    )
    ##############################################################################################################################
    # FDN configurations
    ##############################################################################################################################v
    parser.add_argument(
        "--n_filters", type=int, default=32, help="number of learnable filters K in the frequency enhancement filter (FEF)"
    )
    parser.add_argument(
        "--sampling_freq",
        type=int,
        default=100,
        help="data sampling frequency(Hz). 20 or 100.",
    )
    parser.add_argument(
        "--cutoff_freq",
        type=int,
        default=1,
        help="decomposition cutoff frequency f_c (Hz)",
    )
    parser.add_argument(
        "--denoising_cutoff_freq",
        type=int,
        default=15,
        help="denoising cutoff frequency f_c^dn (Hz)",
    )
    ##############################################################################################################################
    # FDN ablation studies
    ##############################################################################################################################
    parser.add_argument(
        "--disable_dethead",
        action="store_true",
        default=False,
        help="disable the trend head (w/o TrdHead): full-band distribution",
    )
    parser.add_argument(
        "--disable_fpn", action="store_true", help="(unused)"
    )
    parser.add_argument(
        "--disable_freq_enhance",
        action="store_true",
        help="disable the frequency enhancement filter (w/o FEF)",
    )
    parser.add_argument(
        "--disable_moe",
        action="store_true",
        help="disable the gating of the FEF experts (w/o FEF-W; w/o FEF-MoE with --n_filters 1)",
    )
    parser.add_argument(
        "--disable_freq_pass",
        action="store_true",
        help="disable the frequency pass filters (w/o FPF)",
    )
    parser.add_argument(
        "--disable_probhead",
        action="store_true",
        help="disable the residual head (w/o ResHead): full-band pointwise regression",
    )
    parser.add_argument(
        "--disable_imu",
        action="store_true",
        default=False,
        help="(unused)",
    )
    parser.add_argument(
        "--disable_jtorque",
        action="store_true",
        default=False,
        help="exclude the actuation signal u (joint differential pressure) from the inputs",
    )
    parser.add_argument(
        "--freeze_jointq",
        action="store_true",
        default=False,
        help="freeze the pretrained joint-state encoders (linear probing)",
    )
    parser.add_argument(
        "--freeze_imu",
        action="store_true",
        default=False,
        help="(unused)",
    )
    parser.add_argument(
        "--freeze_jtorque",
        action="store_true",
        default=False,
        help="freeze the actuation-signal encoder",
    )
    ##############################################################################################################################
    # Backbone-wise configurations
    ##############################################################################################################################
    # FEDformer
    parser.add_argument(
        "--version",
        type=str,
        default="Fourier",
        help="for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]",
    )
    parser.add_argument(
        "--mode_select",
        type=str,
        default="random",
        help="for FEDformer, there are two mode selection method, options: [random, low]",
    )
    parser.add_argument(
        "--modes", type=int, default=64, help="modes to be selected random 64"
    )
    parser.add_argument("--L", type=int, default=3, help="ignore level")
    parser.add_argument("--base", type=str, default="legendre", help="mwt base")
    parser.add_argument(
        "--cross_activation",
        type=str,
        default="tanh",
        help="mwt cross atention activation function tanh or softmax",
    )
    # parser.add_argument(
    #     "--moving_avg", default=[24], help="window size of moving average") # Fixed, Autoformer=[24], FEDformer=[7, 12, 14, 24, 48]
    ##############################################################################################################################
    # iTransformer
    parser.add_argument(
        "--inverse", action="store_true", help="inverse output data", default=False
    )
    parser.add_argument(
        "--class_strategy",
        type=str,
        default="projection",
        help="projection/average/cls_token",
    )
    ##############################################################################################################################
    # PatchTST
    parser.add_argument("--patch_len", type=int, default=24, help="patch length")
    parser.add_argument(
        "--use_rev_in",
        type=int,
        default=True,
        help="use reversible instance norm in PatchTST and iTransformer",
    )

    # forecasting task
    parser.add_argument(
        "--seq_len", type=int, default=100, help="input sequence length"
    )
    parser.add_argument("--label_len", type=int, default=50, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=100, help="prediction sequence length"
    )
    ##############################################################################################################################
    # RBFNN
    parser.add_argument(
        "--n_kernels", type=int, default=7, help="number of RBF kernels"
    )
    ##############################################################################################################################
    # GPR
    parser.add_argument(
        "--num_inducing", type=int, default=1024, help="number of inducing points"
    )
    parser.add_argument(
        "--num_latents", type=int, default=3, help="number of inducing points"
    )
    ##############################################################################################################################
    # SVM
    parser.add_argument("--svm_kernel", type=str, default="rbf", help="kernel for SVM")
    ##############################################################################################################################
    # Model definition
    ##############################################################################################################################
    parser.add_argument(
        "--enc_in", type=int, default=35, help="encoder input size"
    )  # 7D jointpos + 7D toolpos + 6D imu + jointpos0 + toolpos0 = 34D total
    parser.add_argument("--dec_in", type=int, default=34, help="decoder input size")
    parser.add_argument(
        "--c_out", type=int, default=6, help="output size"
    )  # 6D force/torque output
    parser.add_argument("--d_model", type=int, default=128, help="dimension of model")
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="num of heads. For FEDformer and Autoformer, must be 8.",
    )
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=128 * 4, help="dimension of fcn")
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
    # parser.add_argument(
    #     "--embed",
    #     type=str,
    #     default="timeF",
    #     help="time features encoding, options:[timeF, fixed, learned]",
    # ) # we do not use time feature encoding.
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )
    ##############################################################################################################################
    # Optimization
    ##############################################################################################################################
    parser.add_argument("--train_epochs", type=int, default=5, help="train epochs")
    parser.add_argument(
        "--train_iters",
        type=int,
        default=None,
        help="train iterations. If set, overrides train_epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size of train input data"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="optimizer learning rate"
    )
    parser.add_argument(
        "--lr_scale_pretrained",
        type=float,
        default=1,
        help="lr<-lr*scale for pretrained model params",
    )
    parser.add_argument(
        "--lr_scale_probheads",
        type=float,
        default=1,
        help="lr<-lr*scale for mu/logvar heads",
    )
    parser.add_argument(
        "--loss_scale_trend",
        type=float,
        default=1,
        help="loss_trend<-lr*scale for mu/logvar heads",
    )
    parser.add_argument(
        "--loss_scale_res",
        type=float,
        default=1,
        help="loss_res<-loss_res*scale",
    )
    parser.add_argument(
        "--loss_scale_logvar_reg",
        type=float,
        default=0,
        help="loss_logvar_reg<-loss_logvar_reg*scale",
    )
    parser.add_argument(
        "--enable_lrrt",
        default=False,
        action="store_true",
        help="run learning rate range test before training",
    )
    parser.add_argument(
        "--enable_lr_scheduling",
        default=True,
        action="store_true",
        help="use OneCycleLR scheduling on optimizer",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="use train_ratio*len(training_dataset) samples for training",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="random seed used for training",
    )
    ##############################################################################################################################
    # Accelerated computing & parallelization
    ##############################################################################################################################
    # MacOS
    if platform.system() == "Darwin":
        parser.add_argument(
            "--device", type=str, default="mps", help="device to be trained on"
        )
    else:  # CUDA
        parser.add_argument(
            "--device", type=str, default="cuda", help="device to be trained on"
        )
    parser.add_argument(
        "--disable_compile",
        action="store_true",
        default=True,
        help="disable torch.compile",
    )  # problems exist with multiprocessing
    parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="disable torch.bfloat16 type autocasting",
    )
    parser.add_argument(
        "--num_parallel_runs",
        type=int,
        default=3,
        help="number of maximum parallel runs",
    )
    parser.add_argument(
        "--itr",
        type=int,
        default=3,
        help="experiment iterations per training configuration",
    )
    ##############################################################################################################################
    configs = parser.parse_args()
    if configs.disable_jtorque:
        configs.freeze_jtorque = True
    # configs.lr_scale_probheads = configs.lr / configs.pred_len # performs badly
    if configs.device != "cuda":
        configs.disable_compile = True
        configs.disable_amp = True
    elif not torch.cuda.is_bf16_supported():
        configs.disable_amp = True
    return configs


def event_namer(model, subname: Optional[str] = None):
    model = get_model(model)
    base_dir = f"exp/runs"  # default directory
    event_name = f"{base_dir}/{subname}"  # add subname with time tag
    return event_name


def get_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    elif isinstance(model, nn.parallel.DataParallel):
        return model.module
    elif hasattr(model, "_orig_mod"):
        return model._orig_mod
    elif hasattr(model, "module") and hasattr(model.module, "__class__"):
        return model.module
    return model


def get_name(model):
    return model.__module__.split(".")[-1]


def get_datetime():
    now = datetime.datetime.now()
    return f"{now:%y%m%d-%H%M}"


def get_event_name(dir: str, model: nn.Module, tag: str = ""):
    event_name = f"{dir}/{get_datetime()}_{get_name(model)}_{uuid.uuid4().hex[:8]}"  # {get_datetime()}_
    if tag != "":
        event_name = f"{dir}/{get_datetime()}_{get_name(model)}_{tag}_{uuid.uuid4().hex[:8]}"  # {get_datetime()}_
    return event_name
