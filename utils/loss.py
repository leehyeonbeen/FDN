import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, label, mu_pred, logvar_pred, reduction='mean'):
        """
        label     : (batch_size, seq_len, features)
        mu_pred   : (batch_size, 1, features) or (batch_size, seq_len, features)
        logvar_pred : (batch_size, 1, features) or (batch_size, seq_len, features)
        """
        # var_pred = logvar_pred.exp() + 1e-6
        # const = torch.log(torch.tensor(2.0 * torch.pi))
        logvar_pred = logvar_pred.clamp(min=-10, max=10)
        log_likelihood = -0.5 * (
            (label - mu_pred) ** 2 / (logvar_pred.exp().clamp_min(1e-6)) + logvar_pred
        )
        if reduction == 'mean':
            return -log_likelihood.mean()
        elif reduction == 'sum':
            return -log_likelihood.sum()
        else:
            return -log_likelihood
