import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


"""
Baseline: radial basis function (RBF) neural network, point-to-point estimator.

Maps the current state x'_t = [q, qdot, qddot, u] to the current wrench W_t with configs.n_kernels
Gaussian basis functions (28 in the paper). The RBF layer follows
https://github.com/rssalessio/PytorchRBFLayer

Reference: Z. Chen, F. Huang, W. Sun, J. Gu, B. Yao, RBF-neural-network-based adaptive robust control
for nonlinear bilateral teleoperation manipulators with uncertainty and time delay, IEEE/ASME TMECH,
2020. https://doi.org/10.1109/TMECH.2019.2962081
"""

class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.enc_in = 6 * 4  # 6DoF representations [q, dq, ddq, tau]
        self.c_out = 6  # 6F/T

        self.n_kernels = configs.n_kernels
        self.kernels_centers = nn.Parameter(
            torch.randn(self.n_kernels, self.enc_in)
        )  # equivalent to c_j
        self.log_shapes = nn.Parameter(
            torch.zeros(self.n_kernels)
        )  # equivalent to 1/b_j
        self.weights = nn.Linear(self.n_kernels, self.c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, input_size = x.size()
        x = x.unsqueeze(1)  # (B,1,Cin)
        c = self.kernels_centers.unsqueeze(0).expand(batch_size, -1, -1)  # (B,N,Cin)
        diff = x - c  # B,N,Cin
        norm = torch.norm(diff, dim=-1, p=2)  # (B,N)
        log_shapes = (
            (self.log_shapes).exp().view(1, -1).expand(batch_size, -1)
        )  # strictly positive scaling parameters (B,N)
        norm = norm * log_shapes  # (B,N)
        rbfs = (-norm.pow(2)).exp()  # Gaussian basis (B,N)
        rbfs = rbfs / (1e-6 + rbfs.sum(dim=-1, keepdim=True))  # Normalization (B,N)

        out = self.weights(rbfs)
        return out
