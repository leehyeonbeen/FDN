import torch
import numpy as np
from copy import deepcopy
import math
import torch.nn.functional as F


# Closed-form CRPS of Gaussian predictions, averaged over all elements
class GaussianCRPS:
    def __init__(self):
        self.count = 0
        self.total = 0.0

    @staticmethod
    def gaussian_crps(label, pred_mu, pred_logvar):
        # label, pred_mu, pred_logvar: (N, T, C)
        # This is the exact CRPS of each scalar marginal. It is valid for the
        # MVN models as well: their learned correlations change joint samples,
        # while pred_logvar remains the variance of each marginal.
        std = pred_logvar.exp().sqrt()
        z = (label - pred_mu) / std
        phi = torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        crps = std * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
        return crps

    def update(self, pred_mu, pred_logvar, label):
        # pred_mu, pred_logvar: (N, T, C), label: (N, T, C)
        batch_crps = self.gaussian_crps(label, pred_mu, pred_logvar).sum()
        self.total += batch_crps
        self.count += label.numel()

    def compute(self):
        return self.total / self.count


# MAE, also used as the CRPS of deterministic estimators (the CRPS of a point estimate is its MAE)
class MeanAbsoluteError:
    def __init__(self):
        self.count = 0
        self.total = 0.0

    @staticmethod
    def mae(pred, label):
        # pred, label: (N, T, C)
        return torch.abs(pred - label)

    def update(self, pred, label):
        # pred, label: (N, T, C)
        batch = self.mae(pred, label).sum()
        self.total += batch
        self.count += label.numel()

    def compute(self):
        return self.total / self.count


# Pointwise RMSE, used for the low-frequency pRMSE
class RootMeanSquaredError:
    def __init__(self):
        self.count = 0
        self.total = 0.0

    @staticmethod
    def sse(pred, label):
        # pred, label: (N, T, C)
        return torch.square(pred - label).sum()

    def update(self, pred, label):
        # pred, label: (N, T, C)
        batch = self.sse(pred, label)
        self.total += batch
        self.count += label.numel()

    def compute(self):
        return torch.sqrt(self.total / self.count)


# wRMSE: RMSE between the windowed RMS values (10 steps = 0.1 s) of the prediction and the label.
# For Gaussian predictions, sqrt(mu^2 + sigma^2) is used as the predicted amplitude.
class RMSError:
    def __init__(self):
        self.count = 0
        self.total = 0.0

    @staticmethod
    def compute_rms(pred, label):
        # For (N, T, C) inputs, windows run along the episode axis N for each horizon step
        if pred.ndim == 3:
            pred_rms = compute_windowed_rms(pred.transpose(0, 1)).transpose(0, 1)
            label_rms = compute_windowed_rms(label.transpose(0, 1)).transpose(0, 1)
        elif pred.ndim == 2:
            pred_rms = compute_windowed_rms(pred)
            label_rms = compute_windowed_rms(label)
        else:
            raise ValueError(f"Unsupported pred dimension: {pred.ndim}")
        return pred_rms, label_rms

    def update_gaussian(self, pred_mu, pred_logvar, label):
        pred_mag = torch.sqrt(pred_mu**2 + pred_logvar.exp())
        pred_rms, label_rms = self.compute_rms(pred_mag, label)
        batch = torch.square(pred_rms - label_rms).sum()
        self.total += batch
        self.count += label_rms.numel()

    def update(self, pred, label):
        pred_rms, label_rms = self.compute_rms(pred, label)
        batch = torch.square(pred_rms - label_rms).sum()
        self.total += batch
        self.count += label_rms.numel()

    def compute(self):
        return torch.sqrt(self.total / self.count)


def compute_windowed_rms(x: torch.Tensor, W: int = 10, stride: int = 1):
    """
    RMS over sliding windows of W steps along the time axis.
    x: (L, C) or (B, L, C) -> (Lw, C) or (B, Lw, C), with Lw = (L - W) // stride + 1
    """
    if x.ndim == 2:
        length, channels = x.size()
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        _, length, channels = x.size()
    x2 = x.contiguous().transpose(1, 2) ** 2  # (B, C, L)
    w = torch.ones((channels, 1, W), device=x.device, dtype=x.dtype) / W  # (1,1,W)
    E = F.conv1d(x2, w, stride=stride, padding=0, groups=channels)  # (B,C,Lw)
    if x.ndim == 2:
        return torch.sqrt(E.squeeze(0).T)  # (Lw, C)
    elif x.ndim == 3:
        return torch.sqrt(E.transpose(1, 2))  # (B, Lw, C)
