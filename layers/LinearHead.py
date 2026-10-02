import torch.nn as nn
from layers.Filter import *


# For (B,L,D) shaped embeddings
class FlattenLinearSequence(nn.Module):
    def __init__(self, in_features: int, pred_len: int, c_out: int) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.c_out = c_out
        self.linear_trend = nn.Linear(in_features, pred_len * c_out)
        self.linear_mu = nn.Linear(in_features, c_out)
        self.linear_logvar = nn.Linear(in_features, c_out)

    def forward(self, enc_out: torch.Tensor):
        batch_size, input_length, d_model = enc_out.size()
        enc_out = enc_out.flatten(1)  # (B,Lin*D)
        trend = self.linear_trend(enc_out).reshape(
            batch_size, self.pred_len, self.c_out
        )  # (B,Lout,Cout)
        mu = self.linear_mu(enc_out).unsqueeze(1)  # (B,1,Cout)
        logvar = self.linear_logvar(enc_out).unsqueeze(1)  # (B,1,Cout)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return trend, mu, logvar


# FDN heads: flatten (N, D) per channel, then linear trend, residual mean, and residual log variance
class FlattenLinearPatch(nn.Module):
    def __init__(self, in_features: int, pred_len: int, dropout: float = 0) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.linear_trend = nn.Linear(in_features, pred_len)
        self.linear_mu = nn.Linear(in_features, pred_len)
        self.linear_logvar = nn.Linear(in_features, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out: torch.Tensor):
        batch_size, input_size, num_patches, d_model = enc_out.size()
        enc_out = enc_out.flatten(0, 1).flatten(1)  # (B*C,N*D)
        enc_out = self.dropout(enc_out)
        trend = (
            self.linear_trend(enc_out)
            .reshape(batch_size, input_size, -1)
            .transpose(1, 2)
        )  # (B,T,C)
        mu = (
            self.linear_mu(enc_out).reshape(batch_size, input_size, -1).transpose(1, 2)
        )  # (B,T,C)
        logvar = (
            self.linear_logvar(enc_out)
            .reshape(batch_size, input_size, -1)
            .transpose(1, 2)
        )  # (B,T,C)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return trend, mu, logvar


# PatchTST head: flatten (N, D) per channel, then one linear projection to T steps
class FlattenLinearPatchPlain(nn.Module):
    def __init__(self, in_features: int, pred_len: int, dropout: float = 0) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.linear_trend = nn.Linear(in_features, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out: torch.Tensor):
        batch_size, input_size, num_patches, d_model = enc_out.size()
        enc_out = enc_out.flatten(0, 1).flatten(1)  # (B*C,N*D)
        enc_out = self.dropout(enc_out)
        trend = (
            self.linear_trend(enc_out)
            .reshape(batch_size, input_size, -1)
            .transpose(1, 2)
        )  # (B,T,C)
        return trend


# PatchTST-Gaussian head: flatten (N, D) per channel, then linear mean and log variance
class FlattenLinearPatchPlainGaussian(nn.Module):
    def __init__(self, in_features: int, pred_len: int, dropout: float = 0) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.linear_mu = nn.Linear(in_features, pred_len)
        self.linear_logvar = nn.Linear(in_features, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out: torch.Tensor):
        batch_size, input_size, num_patches, d_model = enc_out.size()
        enc_out = enc_out.flatten(0, 1).flatten(1)  # (B*C,N*D)
        enc_out = self.dropout(enc_out)
        mu = (
            self.linear_mu(enc_out).reshape(batch_size, input_size, -1).transpose(1, 2)
        )  # (B,T,C)
        logvar = (
            self.linear_logvar(enc_out)
            .reshape(batch_size, input_size, -1)
            .transpose(1, 2)
        )  # (B,T,C)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar


# Trend/residual heads for (B, C, D) token embeddings (unused)
class FlattenLinearInverted(nn.Module):
    def __init__(self, in_features: int, pred_len: int, dropout: float = 0) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.linear_trend = nn.Linear(in_features, pred_len)
        self.linear_mu = nn.Linear(in_features, pred_len)
        self.linear_logvar = nn.Linear(in_features, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out: torch.Tensor):
        batch_size, input_size, d_model = enc_out.size()
        enc_out = self.dropout(enc_out)
        trend = self.linear_trend(enc_out).transpose(1, 2)
        mu = self.linear_mu(enc_out).transpose(1, 2)
        logvar = self.linear_logvar(enc_out).transpose(1, 2)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return trend, mu, logvar

