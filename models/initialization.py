import torch
import torch.nn as nn
import math


# Global initialization policy
@torch.no_grad()
def initialize_params(*modules):
    for m in modules:
        for n, p in m.named_parameters():
            n = n.lower()
            if p.numel() < 2:
                continue
            else:
                # normalization layers
                if "norm" in n and "weight" in n:
                    nn.init.ones_(p)
                # FreqEnhance layers
                elif "kernel_weight_real" in n:
                    # nn.init.ones_(p)
                    nn.init.constant_(p, math.log(math.exp(1) - 1))  # softplus==1
                    p.add_(1e-2 * torch.randn_like(p))
                elif "kernel_weight_imag" in n:
                    nn.init.zeros_(p)
                    p.add_(1e-2 * torch.randn_like(p))
                # RBF layers
                elif "kernels_centers" in n:
                    nn.init.uniform_(p, -1.0, 1.0)
                elif "log_shapes" in n:
                    nn.init.zeros_(p)
                # Correlated Gaussian Cholesky parameters: identity correlation
                elif "raw_cholesky" in n:
                    nn.init.zeros_(p)
                    p.diagonal(dim1=-2, dim2=-1).fill_(
                        math.log(math.expm1(1.0))
                    )
                # all bias to zeros
                elif "bias" in n:
                    nn.init.zeros_(p)
                else:
                    nn.init.normal_(p, mean=0, std=1e-2)
