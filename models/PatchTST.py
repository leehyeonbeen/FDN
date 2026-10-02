from layers.MLP import *
from layers.Patch import PatchTSTEncoder
from layers.LinearHead import *
from layers.Filter import *
from layers.utils import *
import torch
import os
import sys

sys.path.append(os.getcwd())


"""
Baseline: PatchTST, sequence-to-sequence estimator.

The original model forecasts only its own input channels, so it is modified for our setting:
a channel-independent patch encoder over the input history x' (24 channels), RevIN inverted on the
encoder representations, a channel-mixing projection to the 6 wrench channels,
and a flatten-linear head over the next T steps.

Reference: Y. Nie, N. H. Nguyen, P. Sinthong, J. Kalagnanam, A time series is worth 64 words:
Long-term forecasting with transformers, ICLR, 2023.
"""

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = 24
        self.c_out = 6
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

        # Modular encoders
        self.encoder = PatchTSTEncoder(
            enc_in=self.enc_in,
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            e_layers=configs.e_layers,
            dropout=configs.dropout,
            n_heads=configs.n_heads,
            patch_len=configs.patch_len,
            activation=configs.activation,
        )
        self.channel_mixer = nn.Linear(self.n_joint * 4, configs.c_out)
        self.decoder = FlattenLinearPatchPlain(
            in_features=self.encoder.num_patches * configs.d_model,
            pred_len=configs.pred_len,
            dropout=configs.dropout,
        )

    def forward(self, x_enc: torch.Tensor):
        batch_size, input_length, input_size = x_enc.size()

        # RevIN norm
        if self.use_rev_in:
            vars, means = torch.var_mean(x_enc, dim=1, keepdim=True)
            stdev = torch.sqrt(vars).clamp_min(1e-6)
            x_enc = (x_enc - means) / stdev

        # Modular sequence encoders (B,L,Cin) -> (B,Cin,P,D)
        enc_out, attns = self.encoder(x_enc)

        # Channel-wise RevIN denorm
        if self.use_rev_in:
            means = means.transpose(1, 2).unsqueeze(3)
            stdev = stdev.transpose(1, 2).unsqueeze(3)
            enc_out = enc_out * stdev + means

        # Apply channel-mixing enc_in -> c_out
        enc_out = enc_out.transpose(1, 3).contiguous()
        enc_out = self.channel_mixer(enc_out)
        enc_out = enc_out.transpose(1, 3).contiguous()

        # Flatten-Linear decoding
        pred = self.decoder(enc_out)  # (B,Lout,Cout)

        # Return results
        if self.output_attention:
            return pred, attns
        else:
            return pred
