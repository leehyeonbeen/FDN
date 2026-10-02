from contextlib import nullcontext
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Correlated Gaussian residuals for the residual correlation study: covariance D R D, where D holds the
# predicted stds and R is a correlation matrix built from learnable Cholesky factors (identity at init).


def _identity_raw_cholesky(*batch_shape: int, size: int) -> torch.Tensor:
    """Return raw parameters whose transformed Cholesky factor is identity."""
    raw = torch.zeros(*batch_shape, size, size)
    raw.diagonal(dim1=-2, dim2=-1).fill_(math.log(math.expm1(1.0)))
    return raw


def _correlation_cholesky(raw_cholesky: torch.Tensor, eps: float) -> torch.Tensor:
    """Map an unconstrained lower triangle to a unit-diagonal correlation factor."""
    diagonal = F.softplus(raw_cholesky.diagonal(dim1=-2, dim2=-1)).clamp_min(eps)
    scale_tril = torch.tril(raw_cholesky, diagonal=-1) + torch.diag_embed(diagonal)

    # If S = LL^T, dividing every row of L by sqrt(diag(S)) makes
    # R = L_corr L_corr^T a positive-definite correlation matrix.
    row_norm = torch.linalg.vector_norm(scale_tril, dim=-1).clamp_min(eps)
    return scale_tril / row_norm.unsqueeze(-1)


def _solve_channel(scale_tril: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Apply L^-1 along the final (channel) axis without batch broadcasting."""
    channels = values.shape[-1]
    original_shape = values.shape
    rhs = values.reshape(-1, channels).transpose(0, 1).contiguous()
    solved = torch.linalg.solve_triangular(scale_tril, rhs, upper=False)
    return solved.transpose(0, 1).reshape(original_shape)


def _multiply_channel(scale_tril: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Apply L along the final (channel) axis without batch broadcasting."""
    channels = values.shape[-1]
    original_shape = values.shape
    flattened = values.reshape(-1, channels)
    transformed = torch.matmul(flattened, scale_tril.transpose(0, 1))
    return transformed.reshape(original_shape)


def _solve_temporal(scale_tril: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Apply L^-1 along the middle (time) axis without batch broadcasting."""
    batch_size, pred_len, channels = values.shape
    rhs = values.permute(1, 0, 2).reshape(pred_len, -1).contiguous()
    solved = torch.linalg.solve_triangular(scale_tril, rhs, upper=False)
    return solved.reshape(pred_len, batch_size, channels).permute(1, 0, 2)


def _multiply_temporal(scale_tril: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Apply L along the middle (time) axis without batch broadcasting."""
    batch_size, pred_len, channels = values.shape
    flattened = values.permute(1, 0, 2).reshape(pred_len, -1).contiguous()
    transformed = torch.matmul(scale_tril, flattened)
    return transformed.reshape(pred_len, batch_size, channels).permute(1, 0, 2)


class _CorrelatedGaussianBase(nn.Module):
    def __init__(self, pred_len: int, c_out: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.c_out = c_out
        self.eps = eps

    def _validate_shapes(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor = None,
    ) -> None:
        expected = (self.pred_len, self.c_out)
        if mu.ndim != 3 or tuple(mu.shape[-2:]) != expected:
            raise ValueError(
                f"Expected mu shape (B, {self.pred_len}, {self.c_out}), "
                f"got {tuple(mu.shape)}."
            )
        if logvar.shape != mu.shape:
            raise ValueError(
                f"Expected logvar shape {tuple(mu.shape)}, got {tuple(logvar.shape)}."
            )
        if target is not None and target.shape != mu.shape:
            raise ValueError(
                f"Expected target shape {tuple(mu.shape)}, got {tuple(target.shape)}."
            )

    @staticmethod
    def _stable_context(reference: torch.Tensor):
        if reference.device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return nullcontext()

    def _standardized_error(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logvar = logvar.float().clamp(min=-10.0, max=10.0)
        standardized = (target.float() - mu.float()) * torch.exp(-0.5 * logvar)
        return standardized, logvar

    def _sample_scale(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logvar = logvar.clamp(max=3.0)
        return mu, torch.exp(0.5 * logvar)


class TemporalCorrelatedGaussian(_CorrelatedGaussianBase):
    """Independent output channels sharing one temporal correlation."""

    def __init__(self, pred_len: int, c_out: int, eps: float = 1e-6) -> None:
        super().__init__(pred_len=pred_len, c_out=c_out, eps=eps)
        self.raw_cholesky_temporal = nn.Parameter(_identity_raw_cholesky(size=pred_len))

    def correlation_cholesky(self) -> torch.Tensor:
        return _correlation_cholesky(self.raw_cholesky_temporal, self.eps)

    def nll(
        self, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        self._validate_shapes(mu, logvar, target)
        with self._stable_context(target):
            standardized, logvar = self._standardized_error(target, mu, logvar)
            scale_tril = self.correlation_cholesky().float()  # (T,T)
            whitened = _solve_temporal(scale_tril, standardized)
            quadratic = whitened.square().sum()
            logdet_temporal = 2.0 * torch.log(scale_tril.diagonal()).sum()
            logdet_corr = self.c_out * logdet_temporal
            event_nll = 0.5 * (quadratic + logvar.sum() + target.shape[0] * logdet_corr)
            return event_nll / target.numel()

    def rsample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        self._validate_shapes(mu, logvar)
        mu, std = self._sample_scale(mu, logvar)
        scale_tril = self.correlation_cholesky().to(dtype=mu.dtype)
        white_noise = torch.randn_like(mu)
        noise = _multiply_temporal(scale_tril, white_noise)
        return mu + std * noise


class ChannelCorrelatedGaussian(_CorrelatedGaussianBase):
    """Independent forecast steps sharing one output-channel correlation."""

    def __init__(self, pred_len: int, c_out: int, eps: float = 1e-6) -> None:
        super().__init__(pred_len=pred_len, c_out=c_out, eps=eps)
        self.raw_cholesky_channel = nn.Parameter(_identity_raw_cholesky(size=c_out))

    def correlation_cholesky(self) -> torch.Tensor:
        return _correlation_cholesky(self.raw_cholesky_channel, self.eps)

    def nll(
        self, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        self._validate_shapes(mu, logvar, target)
        with self._stable_context(target):
            standardized, logvar = self._standardized_error(target, mu, logvar)
            scale_tril = self.correlation_cholesky().float()  # (C,C)
            whitened = _solve_channel(scale_tril, standardized)
            quadratic = whitened.square().sum()
            logdet_corr = 2.0 * torch.log(scale_tril.diagonal()).sum()
            event_nll = 0.5 * (
                quadratic + logvar.sum() + target.shape[0] * self.pred_len * logdet_corr
            )
            return event_nll / target.numel()

    def rsample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        self._validate_shapes(mu, logvar)
        mu, std = self._sample_scale(mu, logvar)
        scale_tril = self.correlation_cholesky().to(dtype=mu.dtype)
        noise = _multiply_channel(scale_tril, torch.randn_like(mu))
        return mu + std * noise


class KroneckerCorrelatedGaussian(_CorrelatedGaussianBase):
    """Separable temporal/channel correlation over the full forecast event."""

    def __init__(self, pred_len: int, c_out: int, eps: float = 1e-6) -> None:
        super().__init__(pred_len=pred_len, c_out=c_out, eps=eps)
        self.raw_cholesky_temporal = nn.Parameter(_identity_raw_cholesky(size=pred_len))
        self.raw_cholesky_channel = nn.Parameter(_identity_raw_cholesky(size=c_out))

    def temporal_cholesky(self) -> torch.Tensor:
        return _correlation_cholesky(self.raw_cholesky_temporal, self.eps)

    def channel_cholesky(self) -> torch.Tensor:
        return _correlation_cholesky(self.raw_cholesky_channel, self.eps)

    def nll(
        self, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        self._validate_shapes(mu, logvar, target)
        with self._stable_context(target):
            standardized, logvar = self._standardized_error(target, mu, logvar)
            temporal_tril = self.temporal_cholesky().float()  # (T,T)
            channel_tril = self.channel_cholesky().float()  # (C,C)

            # W = L_t^{-1} Z L_c^{-T}
            whitened = _solve_temporal(temporal_tril, standardized)
            whitened = _solve_channel(channel_tril, whitened)
            quadratic = whitened.square().sum()

            logdet_temporal = 2.0 * torch.log(temporal_tril.diagonal()).sum()
            logdet_channel = 2.0 * torch.log(channel_tril.diagonal()).sum()
            logdet_corr = self.c_out * logdet_temporal + self.pred_len * logdet_channel
            event_nll = 0.5 * (quadratic + logvar.sum() + target.shape[0] * logdet_corr)
            return event_nll / target.numel()

    def rsample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        self._validate_shapes(mu, logvar)
        mu, std = self._sample_scale(mu, logvar)
        temporal_tril = self.temporal_cholesky().to(dtype=mu.dtype)
        channel_tril = self.channel_cholesky().to(dtype=mu.dtype)
        noise = _multiply_temporal(temporal_tril, torch.randn_like(mu))
        noise = _multiply_channel(channel_tril, noise)
        return mu + std * noise
