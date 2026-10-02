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
FDN for transfer learning with relative position inputs.

Same as FDN_PatchTST_RelPos6D with n = 7 joints (35 inputs) to cover the 6- and 7-DoF robots
in RH20T; 7th-joint channels are zero for 6-DoF data. Used for RH20T pretraining (actuation
channels disabled) and for linear probing / fine-tuning on the hydraulic dataset.
"""

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = 35
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

        self.n_joint = 7  # J1~J7

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
        # One PatchTST encoder per modality (dq, qdot, qddot, u)
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
        # MLP encoder for the initial position q0 (added to the dq features)
        self.encoder_jointpos_0 = LayerNormMLP(
            self.n_joint,
            configs.d_model,
            configs.d_model,
            dropout=configs.dropout,
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
                self.encoder_jointpos_0,
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
        res = sample_gaussian(mu, logvar, self.pred_len, std_gain=1.0) # std_gain
        # if not self.disable_freq_pass:
        #     res = self.freq_pass_high(res)
        return res

    # x_enc: (B, L, C_in) input history; channel_mask: (B, C_in), True for valid channels.
    # Returns the trend, sampled residual, residual mean, and residual log variance, each (B, T, 6).
    def forward(self, x_enc: torch.Tensor, channel_mask: torch.Tensor = None):
        batch_size, input_length, input_size = x_enc.size()

        # channel_mask: using channels -> True, else -> False
        # Ex. 6DoF representations -> 7th joint channels = False
        # Ex. No jtorque data -> jtorque channels = False
        if channel_mask is None:
            channel_mask = torch.ones(
                (batch_size, input_size), dtype=torch.bool, device=x_enc.device
            )

        # Zero out invalid input channels
        mask_input = channel_mask.unsqueeze(1).expand(-1, input_length, -1)
        x_enc = x_enc.masked_fill(~mask_input, 0.0)

        # Split input channels by modality (n channels each)
        idx_1 = self.n_joint  # jointpos
        idx_2 = idx_1 + self.n_joint  # jointvel
        idx_3 = idx_2 + self.n_joint  # jointacc
        idx_4 = idx_3 + self.n_joint  # jtorque
        idx_5 = idx_4 + self.n_joint  # jointpos_0

        jointpos = x_enc[:, :, :idx_1]
        jointvel = x_enc[:, :, idx_1:idx_2]
        jointacc = x_enc[:, :, idx_2:idx_3]
        jtorque = x_enc[:, :, idx_3:idx_4]
        jointpos_0 = x_enc[:, :, idx_4:idx_5]

        # Ignore u (zeroed), e.g. for RH20T pretraining
        if self.disable_jtorque:
            jtorque = jtorque * 0.0

        # Time-varying inputs [dq, qdot, qddot, u] -> (B, L, 4n)
        x_delta = torch.cat([jointpos, jointvel, jointacc, jtorque], dim=-1)

        # RevIN (masked channels are left as zeros)
        if self.use_rev_in:
            rev_in_mask = channel_mask[:, :idx_4].unsqueeze(1)
            vars, means = torch.var_mean(x_delta, dim=1, keepdim=True)
            stdev = torch.sqrt(vars).clamp_min(1e-6)
            means = means.masked_fill(~rev_in_mask, 0.0)
            stdev = stdev.masked_fill(~rev_in_mask, 1.0)
            x_delta = (x_delta - means) / stdev

        # FEF: frequency-domain filtering of each input channel
        if not self.disable_freq_enhance:
            x_delta = self.freq_enhance(x_delta)

        # Re-apply the channel mask after FEF
        mask_x_delta = mask_input[..., :idx_4]
        x_delta = x_delta.masked_fill(~mask_x_delta, 0.0)

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

        # Add the q0 embedding to the dq features
        ic_jointpos = self.encoder_jointpos_0(jointpos_0[:, 0:1, :]).unsqueeze(1)
        ic_jointpos = ic_jointpos.expand(-1, -1, self.encoder_jointpos.num_patches, -1)
        enc_out = torch.cat(
            [enc_out[:, :idx_1, ...] + ic_jointpos, enc_out[:, idx_1:, ...]],
            dim=1,
        )

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
