import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.signal import savgol_coeffs, butter
from layers.MLP import *
import numpy as np
import math
import torchaudio.functional as T


# FEF: K = n_filters learnable magnitude filters on the input spectrum, mixed by input-dependent
# softmax gates. Only magnitudes change, so phase and shape are kept. Without MoE, filters are averaged.
class FreqEnhanceFilter(nn.Module):
    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        n_filters: int = 1,
        disable_moe: bool = False,
    ) -> None:
        super().__init__()
        self.enc_in = enc_in
        self.freq_len = seq_len // 2 + 1
        self.seq_len = seq_len
        self.n_filters = n_filters
        self.disable_moe = disable_moe

        if self.n_filters > 1 and not disable_moe:
            self.gating = nn.Linear(seq_len * enc_in, n_filters)  # time-domain gating
            # self.gating = nn.Linear((seq_len // 2 + 1) * enc_in, n_filters) # freq-domain gating
        # Filter weights per frequency bin, channel, and filter; softplus keeps them positive
        self.kernel_weight_real = nn.Parameter(
            torch.ones(1, seq_len // 2 + 1, enc_in, n_filters)
        )

    def forward(self, x, output_internals: bool = False):
        batch_size, input_length, input_size = x.size()
        x = x.to(torch.float32)  # ensure FP32
        x = x.unsqueeze(-1)  # (B,L,C,1)

        # Gating weights from the flattened time-domain input (uniform without MoE)
        if self.n_filters > 1 and not self.disable_moe:
            # weight_gating = F.softmax(self.gating(x), dim=-1)
            weight_gating = F.softmax(self.gating(x.flatten(1)), dim=-1)  # Linear
            weight_gating = weight_gating.contiguous().view(
                batch_size, 1, 1, self.n_filters
            )
        else:
            weight_gating = torch.full(
                (batch_size, 1, 1, self.n_filters),
                1.0 / self.n_filters,
                device=x.device,
            )

        # Scale the spectrum by each filter (magnitude only), then mix the filters with the gates
        fft = torch.fft.rfft(x, dim=1)  # (fft_real + fft_imag)
        fft_real = fft.real
        fft_imag = fft.imag

        weight_filter = F.softplus(self.kernel_weight_real)
        filtered_fft_real = fft_real * weight_filter
        filtered_fft_imag = fft_imag * weight_filter
        filtered_fft = torch.complex(filtered_fft_real, filtered_fft_imag)
        filtered_fft = (filtered_fft * weight_gating).sum(dim=-1)

        x = torch.fft.irfft(
            filtered_fft,
            n=self.seq_len,
            dim=1,
        )

        # Time domain weighted sum
        # x = (x * weight_gating).sum(dim=-1)

        if output_internals:
            return x, weight_gating, filtered_fft
        else:
            return x


# FPF: zero-phase filter that multiplies the FFT by a real Butterworth magnitude response.
# 'low': low-pass at cutoff_freq. 'high': high-pass at cutoff_freq, or band-pass up to
# denoising_cutoff_freq. Used for the trend/residual decomposition, the output filters, and denoising.
class FreqPassFilter(nn.Module):
    def __init__(
        self,
        mode: str = "low",
        cutoff_freq: int = 1,
        sampling_freq: int = 100,
        denoising_cutoff_freq: int = 0,  # 15 Hz for model predictions only
        pad: str = "both",
        order: int = 8,
    ) -> None:
        super().__init__()
        assert mode.lower() in [
            "low",
            "high",
        ], "Filtering mode should be entered properly."
        assert pad in [
            "none",
            "pre",
            "both",
        ], "Padding mode should be entered properly."
        if denoising_cutoff_freq > 0:
            assert denoising_cutoff_freq > cutoff_freq
        self.mode = mode.lower()
        self.cutoff_freq = cutoff_freq
        self.sampling_freq = sampling_freq
        self.order = order
        self.pad = pad
        self.denoising_cutoff_freq = denoising_cutoff_freq

    def filter(self, x, mode: int = 1):
        batch_size, length, output_size = x.size()
        x = x.transpose(1, 2)
        kernel = butter_response(
            length,
            self.cutoff_freq,
            self.sampling_freq,
            self.order,
            device=x.device,
        )

        # Low-pass response H; 1 - H gives the high-pass, and multiplying by the denoising low-pass gives the band-pass
        if self.mode == "high":
            kernel = 1.0 - kernel  # high-pass
            if self.denoising_cutoff_freq > 0:
                kernel_denoise = butter_response(
                    length,
                    self.denoising_cutoff_freq,
                    self.sampling_freq,
                    self.order,
                    device=x.device,
                )
                kernel = (
                    kernel * kernel_denoise
                )  # band-pass [cutoff_freq, denoising_cutoff_freq]

        fft = torch.fft.rfft(x.to(torch.float32), dim=-1)
        filtered_fft_real = fft.real * kernel
        filtered_fft_imag = fft.imag * kernel
        fft = torch.complex(filtered_fft_real, filtered_fft_imag)
        x = torch.fft.irfft(fft, n=length, dim=-1)
        x = x.transpose(1, 2)
        return x

    def forward(self, x, mode: int = 1):
        isnumpy = isinstance(x, np.ndarray)
        if isnumpy:
            x = torch.from_numpy(x)
        batched = True if x.ndim == 3 else False
        x = x.unsqueeze(0) if not batched else x

        batch_size, length, output_size = x.size()
        padding = length // 2
        # Reflection padding to reduce boundary artifacts
        if self.pad == "both":
            x = F.pad(x, (0, 0, padding, padding), mode="reflect")  # padding
        elif self.pad == "pre":
            x = F.pad(x, (0, 0, padding, 0), mode="reflect")  # pre-padding

        # filter
        x = self.filter(x, mode=mode)

        # remove padding
        if self.pad in ["both", "pre"]:
            x = x[:, padding : padding + length, :]  # remove pad

        if not batched:
            x = x.squeeze(0)
        if isnumpy:
            x = x.detach().cpu().numpy()
        return x


class CausalLPF_Butter(nn.Module):
    # Causal Butterworth IIR low-pass filter (stateful, second-order sections).
    # Used in preprocessing to denoise joint positions and actuation signals without look-ahead.
    def __init__(
        self,
        cutoff_freq: float = 15.0,
        sampling_freq: float = 100.0,
        order: int = 4,
    ):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.sampling_freq = sampling_freq
        self.order = order
        sos = butter(
            order,
            cutoff_freq,
            btype="low",
            analog=False,
            fs=sampling_freq,
            output="sos",
        )
        sos = torch.from_numpy(sos)
        self.register_buffer("sos", sos)

    def init_state(self, batch_size: int, n_channels: int, device, dtype):
        n_sec = self.sos.size(0)
        # (n_sections, B, C, 2)  -> DF2T state z1,z2
        return torch.zeros(n_sec, batch_size, n_channels, 2, device=device, dtype=dtype)

    def filter(self, x, state=None, return_state: bool = False):
        # x: (B,L,C)
        batch_size, length, n_ch = x.size()
        y = x

        sos = self.sos.to(device=x.device, dtype=x.dtype)
        if state is None:
            state = self.init_state(batch_size, n_ch, x.device, x.dtype)
        else:
            state = state.to(device=x.device, dtype=x.dtype)

        # process SOS sections
        # DF2T per section:
        # y = b0*x + z1
        # z1 = b1*x - a1*y + z2
        # z2 = b2*x - a2*y
        for s in range(sos.size(0)):
            b0, b1, b2, a0, a1, a2 = sos[s]
            if a0.abs() > 0:
                b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
                a1, a2 = a1 / a0, a2 / a0

            z1 = state[s, :, :, 0]
            z2 = state[s, :, :, 1]

            out = torch.empty_like(y)
            for t in range(length):
                xt = y[:, t, :]  # (B,C)
                yt = b0 * xt + z1
                z1_new = b1 * xt - a1 * yt + z2
                z2_new = b2 * xt - a2 * yt
                out[:, t, :] = yt
                z1, z2 = z1_new, z2_new

            y = out
            state[s, :, :, 0] = z1
            state[s, :, :, 1] = z2

        if return_state:
            return y, state
        return y

    def forward(self, x, state=None, return_state: bool = False):
        isnumpy = isinstance(x, np.ndarray)
        if isnumpy:
            x = torch.from_numpy(x)

        # normalize shape -> (B,L,C)
        if x.ndim == 1:  # (C,)
            x = x.unsqueeze(0).unsqueeze(0)
            batched = False
            single_step = True
        elif x.ndim == 2:  # (L,C)
            x = x.unsqueeze(0)
            batched = False
            single_step = False
        else:  # (B,L,C)
            batched = True
            single_step = False

        y = self.filter(x, state=state, return_state=return_state)

        if return_state:
            y, state = y

        if single_step:
            y = y.squeeze(0).squeeze(0)  # (C,)
        elif not batched:
            y = y.squeeze(0)  # (L,C)

        if isnumpy:
            y = y.detach().cpu().numpy()
            if return_state:
                state = state.detach().cpu().numpy()
        if return_state:
            return y, state
        return y


class CausalDiff_SavGol(FreqPassFilter):
    # Causal Savitzky-Golay differentiation: velocity and acceleration at the end of each trailing window.
    # Used in preprocessing. Needs window_length - 1 past samples; the episode start is replicate-padded.
    def __init__(
        self,
        cutoff_freq: float = 15.0,
        sampling_freq: float = 100.0,
        pad="none",
        window_length: int = 31,
        polyorder: int = 3,
    ):
        # assert window_length % 2 == 1, "Window length should be odd."
        assert polyorder < window_length, "polyorder must be < window_length."

        super().__init__(cutoff_freq=cutoff_freq, sampling_freq=sampling_freq, pad=pad)
        self.window_length = window_length
        self.polyorder = polyorder
        kernel_d = (
            torch.FloatTensor(
                savgol_coeffs(
                    window_length=window_length,
                    polyorder=polyorder,
                    deriv=1,
                    delta=1 / sampling_freq,
                    use="dot",
                    pos=self.window_length - 1,  # causal & no delay
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        kernel_dd = (
            torch.FloatTensor(
                savgol_coeffs(
                    window_length=window_length,
                    polyorder=polyorder,
                    deriv=2,
                    delta=1 / sampling_freq,
                    use="dot",
                    pos=self.window_length - 1,  # causal & no delay
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.register_buffer("kernel_d", kernel_d)
        self.register_buffer("kernel_dd", kernel_dd)

    def filter(self, x, mode: int = 1):
        batch_size, length, output_size = x.size()
        x = x.transpose(1, 2)

        # initial window padding
        x = F.pad(x, (self.window_length - 1, 0), mode="replicate")

        # differentiation kernels
        if mode == 1:
            kernel = self.kernel_d.to(x.dtype).expand(output_size, -1, -1)
        elif mode == 2:
            kernel = self.kernel_dd.to(x.dtype).expand(output_size, -1, -1)
        dx = F.conv1d(x, kernel, bias=None, stride=1, padding=0, groups=output_size)
        return dx.transpose(1, 2)

    def vel_acc(self, x):
        vel = self.forward(x, mode=1)
        acc = self.forward(x, mode=2)
        return vel, acc


class CausalLPF_EMA(FreqPassFilter):
    # Causal first-order low-pass filter (EMA). Alternative to CausalLPF_Butter, not used.
    def __init__(
        self,
        cutoff_freq: float = 15.0,
        sampling_freq: float = 100.0,
        pad="none",
        n_cascade: int = 1,
    ):
        super().__init__(cutoff_freq=cutoff_freq, sampling_freq=sampling_freq, pad=pad)
        self.n_cascade = n_cascade
        self.beta = math.exp(-2.0 * math.pi * cutoff_freq / sampling_freq)

    def filter(self, x, mode: int = 1):
        batch_size, length, output_size = x.size()
        x = x.transpose(0, 1)
        for _ in range(self.n_cascade):
            out = torch.empty_like(x)
            out[0] = x[0]
            for t in range(1, length):
                out[t] = self.beta * out[t - 1] + (1.0 - self.beta) * x[t]
            x = out
        x = x.transpose(0, 1).contiguous()
        return x


class CausalDiff_DD(FreqPassFilter):
    # Causal dirty derivative. Alternative to CausalDiff_SavGol without a history window, not used.
    def __init__(
        self, cutoff_freq: float = 15.0, sampling_freq: float = 100.0, pad="none"
    ):
        super().__init__(cutoff_freq=cutoff_freq, sampling_freq=sampling_freq, pad=pad)
        dt = 1.0 / sampling_freq
        tau = 1.0 / (2.0 * math.pi * cutoff_freq)
        alpha = (2.0 * tau - dt) / (2.0 * tau + dt)
        beta = 2.0 / (2.0 * tau + dt)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))

    def vel_acc(self, x):
        vel = self.forward(x)
        acc = self.forward(vel)
        return vel, acc

    def filter(self, x, mode: int = 1):
        batch_size, length, output_size = x.size()
        x = x.transpose(0, 1)
        dx = torch.zeros_like(x)
        x_prev = x[0]
        dx_prev = dx[0]
        for t in range(1, length):
            dx[t] = self.alpha * dx_prev + self.beta * (x[t] - x_prev)
            x_prev = x[t]
            dx_prev = dx[t]
        dx = dx.transpose(0, 1).contiguous()
        return dx


# Real Butterworth magnitude response 1 / sqrt(1 + (f / fc)^(2 * order)) on the rFFT bins, (1, 1, L//2+1)
def butter_response(
    seq_len,
    cutoff_freq: int = 1,
    sampling_freq: float = 100.0,
    filt_order: int = 4,
    device: str = "cpu",
):
    freq = torch.fft.rfftfreq(n=seq_len, d=1 / sampling_freq, device=device)
    ratio = freq / (cutoff_freq + 1e-6)
    kernel = 1.0 / torch.sqrt(1.0 + ratio ** (2 * filt_order))
    kernel = kernel.reshape(1, 1, -1)  # (1,1,L//2+1)
    return kernel


# Std gain that keeps unit variance after filtering white noise with the high-/band-pass response (unused)
def get_std_gain(
    pred_len: int = 100,
    cutoff_freq: int = 1,
    sampling_freq: int = 100,
    denoising_cutoff_freq: int = 15,
):
    kernel_lp = butter_response(
        pred_len,
        cutoff_freq,
        sampling_freq,
        device="cpu",
    )
    kernel_hp = 1.0 - kernel_lp
    if denoising_cutoff_freq > 0:
        kernel_denoise = butter_response(
            pred_len, denoising_cutoff_freq, sampling_freq, device="cpu"
        )
        kernel_hp = kernel_hp * kernel_denoise
    power = (kernel_hp**2).mean(dim=-1, keepdim=True)
    gain = 1 / torch.sqrt(power + 1e-12)
    return gain.item()
