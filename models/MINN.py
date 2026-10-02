import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Baseline: model-independent neural network (MINN), point-to-point estimator.

An MLP with GELU activations that maps the current state x'_t = [q, qdot, qddot, u] (24 inputs)
to the current wrench W_t, following the Type II input configuration (q, qdot, qddot, tau -> F_ext),
which performed best in the original work.

Reference: A. C. Smith, F. Mobasser, K. Hashtrudi-Zaad, Neural-network-based contact force observers
for haptic applications, IEEE T-RO, 2006. https://doi.org/10.1109/TRO.2006.882923
"""

class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.enc_in = 6 * 4  # 6DoF representations [q, dq, ddq, tau]
        self.c_out = 6  # 6F/T
        self.num_hidden_layers = configs.e_layers
        self.mlp = []
        self.mlp.append(nn.Linear(self.enc_in, configs.d_model))
        self.mlp.append(nn.GELU())
        for i in range(configs.e_layers - 1):
            self.mlp.append(nn.Linear(configs.d_model, configs.d_model))
            self.mlp.append(nn.GELU())
        self.mlp.append(nn.Linear(configs.d_model, self.c_out))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        return out
