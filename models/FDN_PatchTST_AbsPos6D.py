from layers.MLP import *
from layers.Patch import PatchTSTEncoder
from layers.LinearHead import FlattenLinearPatch
from layers.Filter import *
from layers.utils import *
import torch
import os
import sys

sys.path.append(os.getcwd())


"""
FDN trained from scratch with absolute position inputs.

Same as FDN_PatchTST_RelPos6D, but takes absolute joint positions q instead of [dq, q0];
the initial-position encoder is removed (24 inputs = 4 modalities x 6 joints).
"""

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = 24
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.use_rev_in = configs.use_rev_in
        self.output_attention = configs.output_attention
        self.disable_freq_enhance = configs.disable_freq_enhance
        self.disable_freq_pass = configs.disable_freq_pass
        self.disable_moe = configs.disable_moe
        self.disable_probhead = configs.disable_probhead
        self.disable_jtorque = configs.disable_jtorque
        self.freeze_jtorque = configs.freeze_jtorque
        self.freeze_jointq = configs.freeze_jointq
        self.cutoff_freq = configs.cutoff_freq
        self.sampling_freq = configs.sampling_freq
        self.denoising_cutoff_freq = configs.denoising_cutoff_freq
        self.disable_dethead = configs.disable_dethead

        self.n_joint = 6  # J1~J6

        # FEF: n_filters learnable frequency filters mixed by input-dependent gates
        self.freq_enhance = FreqEnhanceFilter(
            self.n_joint * 4,
            configs.seq_len,
            configs.n_filters,
            disable_moe=configs.disable_moe,
        )
        # FPF: low-pass for the trend (< 1 Hz), band-pass for the residual mean (1-15 Hz)
        self.freq_pass_low = FreqPassFilter(
            mode="low",
            cutoff_freq=configs.cutoff_freq,
            sampling_freq=configs.sampling_freq,
            denoising_cutoff_freq=configs.denoising_cutoff_freq,
        )
        self.freq_pass_high = FreqPassFilter(
            mode="high",
            cutoff_freq=configs.cutoff_freq,
            sampling_freq=configs.sampling_freq,
            denoising_cutoff_freq=configs.denoising_cutoff_freq,
        )
        # One PatchTST encoder per modality (q, qdot, qddot, u)
        self.encoder_jointpos = PatchTSTEncoder(
            enc_in=self.n_joint,
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            e_layers=configs.e_layers,
            dropout=configs.dropout,
            n_heads=configs.n_heads,
            patch_len=configs.patch_len,
            activation=configs.activation,
        )
        self.encoder_jointvel = PatchTSTEncoder(
            enc_in=self.n_joint,
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            e_layers=configs.e_layers,
            dropout=configs.dropout,
            n_heads=configs.n_heads,
            patch_len=configs.patch_len,
            activation=configs.activation,
        )
        self.encoder_jointacc = PatchTSTEncoder(
            enc_in=self.n_joint,
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            e_layers=configs.e_layers,
            dropout=configs.dropout,
            n_heads=configs.n_heads,
            patch_len=configs.patch_len,
            activation=configs.activation,
        )
        self.encoder_jtorque = PatchTSTEncoder(
            enc_in=self.n_joint,
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            e_layers=configs.e_layers,
            dropout=configs.dropout,
            n_heads=configs.n_heads,
            patch_len=configs.patch_len,
            activation=configs.activation,
        )

        # Channel mixer: 4n input channels -> 6 wrench channels
        self.channel_mixer = nn.Linear(self.n_joint * 4, configs.c_out)
        # Asymmetric heads: pointwise trend, Gaussian residual (mean, log variance)
        self.decoder = FlattenLinearPatch(
            in_features=self.encoder_jointpos.num_patches * configs.d_model,
            pred_len=configs.pred_len,
            dropout=configs.dropout,
        )

        # Freeze modules for transfer learning and ablations
        if self.freeze_jtorque:
            for p in self.encoder_jtorque.parameters():
                p.requires_grad = False
        if self.freeze_jointq:
            for encoder in [
                self.encoder_jointpos,
                self.encoder_jointvel,
                self.encoder_jointacc,
            ]:
                for p in encoder.parameters():
                    p.requires_grad = False
        if self.disable_freq_enhance:
            for p in self.freq_enhance.parameters():
                p.requires_grad = False
        # w/o ResHead: the trend head covers the full band, so its filter becomes a 15 Hz low-pass
        if self.disable_probhead:
            for p in self.decoder.linear_mu.parameters():
                p.requires_grad = False
            for p in self.decoder.linear_logvar.parameters():
                p.requires_grad = False
            self.freq_pass_low = FreqPassFilter(
                mode="low",
                cutoff_freq=configs.denoising_cutoff_freq,
                sampling_freq=configs.sampling_freq,
                denoising_cutoff_freq=0,
            )  # filter trend(=full prediction) with 15Hz LPF
        # w/o TrdHead: the residual head covers the full band, so its filter becomes a 15 Hz low-pass
        if self.disable_dethead:
            for p in self.decoder.linear_trend.parameters():
                p.requires_grad = False
            self.freq_pass_high = FreqPassFilter(
                mode="low",
                cutoff_freq=configs.denoising_cutoff_freq,
                sampling_freq=configs.sampling_freq,
                denoising_cutoff_freq=0,
            )  # filter res(=full prediction) with 15Hz LPF

    # Std correction for band-pass filtering the sampled residual.
    # Unused: the sampled residual is not filtered, so sample_residual uses std_gain=1.0.
    def get_std_gain(self):
        if self.disable_freq_pass:
            std_gain = 1.0
        elif self.disable_dethead:
            std_gain = get_std_gain(
                self.pred_len,
                cutoff_freq=self.denoising_cutoff_freq,
                sampling_freq=self.sampling_freq,
                denoising_cutoff_freq=0,
            )
        else:
            std_gain = get_std_gain(
                self.pred_len,
                cutoff_freq=self.cutoff_freq,
                sampling_freq=self.sampling_freq,
                denoising_cutoff_freq=self.denoising_cutoff_freq,
            )
        return std_gain

    # Residual sample: mean + std * N(0, 1), independent over steps and channels
    def sample_residual(self, mu, logvar):
        # std_gain = self.get_std_gain()
        res = sample_gaussian(mu, logvar, self.pred_len, std_gain=1.0)
        # if not self.disable_freq_pass:
            # res = self.freq_pass_high(res)
        return res

    # x_enc: (B, L, C_in) input history; channel_mask: (B, C_in), True for valid channels.
    # Returns the trend, sampled residual, residual mean, and residual log variance, each (B, T, 6).
    def forward(self, x_enc: torch.Tensor, channel_mask: torch.Tensor = None):
        batch_size, input_length, input_size = x_enc.size()

        # Split input channels by modality (n channels each)
        idx_1 = self.n_joint  # jointpos
        idx_2 = idx_1 + self.n_joint  # jointvel
        idx_3 = idx_2 + self.n_joint  # jointacc
        idx_4 = idx_3 + self.n_joint  # jtorque

        jointpos = x_enc[:, :, :idx_1]
        jointvel = x_enc[:, :, idx_1:idx_2]
        jointacc = x_enc[:, :, idx_2:idx_3]
        jtorque = x_enc[:, :, idx_3:idx_4]

        # Ignore u (zeroed), e.g. for RH20T pretraining
        if self.disable_jtorque:
            jtorque = jtorque * 0.0

        # Inputs [q, qdot, qddot, u] (absolute positions) -> (B, L, 4n)
        x_delta = torch.cat([jointpos, jointvel, jointacc, jtorque], dim=-1)

        # RevIN
        if self.use_rev_in:
            vars, means = torch.var_mean(x_delta, dim=1, keepdim=True)
            stdev = torch.sqrt(vars).clamp_min(1e-6)
            x_delta = (x_delta - means) / stdev

        # FEF: frequency-domain filtering of each input channel
        if not self.disable_freq_enhance:
            x_delta = self.freq_enhance(x_delta)

        # Split the enhanced inputs back into modalities
        jointpos = x_delta[:, :, :idx_1]
        jointvel = x_delta[:, :, idx_1:idx_2]
        jointacc = x_delta[:, :, idx_2:idx_3]
        jtorque = x_delta[:, :, idx_3:idx_4]

        # Ignore u (zeroed), e.g. for RH20T pretraining
        if self.disable_jtorque:
            jtorque = jtorque * 0.0

        # Encode each modality: (B,L,n) -> (B,n,N,D); zeros for u if ignored
        enc_out_jointpos, attns_jointpos = self.encoder_jointpos(jointpos)
        enc_out_jointvel, attns_jointvel = self.encoder_jointvel(jointvel)
        enc_out_jointacc, attns_jointacc = self.encoder_jointacc(jointacc)
        if not self.disable_jtorque:
            enc_out_jtorque, attns_jtorque = self.encoder_jtorque(jtorque)
        else:
            enc_out_jtorque = torch.zeros(
                batch_size,
                self.n_joint,
                self.encoder_jointpos.num_patches,
                self.d_model,
                device=x_enc.device,
                dtype=x_enc.dtype,
            )

        # Stack along channels -> (B,4n,N,D)
        enc_out = torch.cat(
            [enc_out_jointpos, enc_out_jointvel, enc_out_jointacc, enc_out_jtorque],
            dim=1,
        )

        # Undo RevIN on the encoder outputs
        if self.use_rev_in:
            means = means.transpose(1, 2).unsqueeze(3)
            stdev = stdev.transpose(1, 2).unsqueeze(3)
            enc_out = enc_out * stdev + means

        # Mix 4n input channels into 6 wrench channels -> (B,6,N,D)
        enc_out = enc_out.transpose(1, 3).contiguous()
        enc_out = self.channel_mixer(enc_out)
        enc_out = enc_out.transpose(1, 3).contiguous()

        # Asymmetric heads: trend, residual mean, residual log variance, each (B,T,6)
        trend, mu, logvar = self.decoder(enc_out)  # (B,Lout,Cout)

        # FPF on both outputs, then sample the residual
        if not self.disable_freq_pass:
            trend = self.freq_pass_low(trend)
            mu = self.freq_pass_high(mu)
        res = self.sample_residual(mu, logvar)

        # Ablated heads return zeros
        if self.output_attention:
            return (trend, res, mu, logvar), (
                attns_jointpos,
                attns_jointvel,
                attns_jointacc,
                attns_jtorque,
            )
        else:
            if self.disable_dethead:
                trend = torch.zeros_like(trend)
                return trend, res, mu, logvar
            elif self.disable_probhead:
                res = torch.zeros_like(res)
                mu = torch.zeros_like(mu)
                logvar = torch.zeros_like(logvar)
                return trend, res, mu, logvar
            else:
                return trend, res, mu, logvar
