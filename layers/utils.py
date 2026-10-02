from sympy import Poly, legendre, Symbol, chebyshevt
from scipy.special import eval_legendre
from functools import partial
import numpy as np
from layers.MLP import MLP
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys

sys.path.append(os.getcwd())


# Gaussian sample mu + std * N(0, 1) over output_length steps, with the log variance clamped at logvar_clamp_max
def sample_gaussian(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    output_length: int,
    logvar_clamp_max: float = 3.0,
    std_gain: float = 1.0,
):
    batch_size, _, output_size = mu.size()  # (B, 1, Cout) or (B, Lout, Cout)
    batch_size, _, output_size = logvar.size()  # (B, 1, Cout) or (B, Lout, Cout)

    logvar = logvar.clamp(max=logvar_clamp_max)
    std = logvar.mul(0.5).exp() * std_gain
    res = (
        torch.randn(
            batch_size, output_length, output_size, device=mu.device, dtype=mu.dtype
        )
        * std
        + mu
    )  # (B, Lout, Cout)
    return res  # (B, Lout, Cout)
