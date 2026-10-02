import torch

from layers.Filter import FreqEnhanceFilter
from models.FDN_PatchTST_RelPos6D import Model as RelPos6DModel


"""
Input ablation of FDN.

FDN_PatchTST_RelPos6D with the selected input modalities and their encoders physically removed
(exclude_jointpos, exclude_jointvel, exclude_jointacc, disable_jtorque). Used with the
input-ablation datasets in data/dataset_ablation.py.
"""

class Model(RelPos6DModel):
    def __init__(self, configs):
        # Modalities to remove; at least one must be excluded
        self.exclude_jointpos = getattr(configs, "exclude_jointpos", False)
        self.exclude_jointvel = getattr(configs, "exclude_jointvel", False)
        self.exclude_jointacc = getattr(configs, "exclude_jointacc", False)
        self.exclude_jtorque = getattr(configs, "disable_jtorque", False)
        if not (
            self.exclude_jointpos
            or self.exclude_jointvel
            or self.exclude_jointacc
            or self.exclude_jtorque
        ):
            raise ValueError(
                "InputAblation requires at least one excluded input modality."
            )

        # Build the full FDN first, then delete the encoders of the excluded modalities
        super().__init__(configs)
        self.num_patches = self.encoder_jointpos.num_patches

        self.input_modalities = []
        if not self.exclude_jointpos:
            self.input_modalities.append("jointpos")
        if not self.exclude_jointvel:
            self.input_modalities.append("jointvel")
        if not self.exclude_jointacc:
            self.input_modalities.append("jointacc")
        if not self.exclude_jtorque:
            self.input_modalities.append("jtorque")

        if self.exclude_jointpos:
            del self.encoder_jointpos
            del self.encoder_jointpos_0
        if self.exclude_jointvel:
            del self.encoder_jointvel
        if self.exclude_jointacc:
            del self.encoder_jointacc
        if self.exclude_jtorque:
            del self.encoder_jtorque

        # Rebuild FEF and the channel-mixing layer for the reduced number of input channels
        n_dynamic_channels = self.n_joint * len(self.input_modalities)
        n_initial_channels = 0 if self.exclude_jointpos else self.n_joint
        self.enc_in = n_dynamic_channels + n_initial_channels
        self.freq_enhance = FreqEnhanceFilter(
            n_dynamic_channels,
            configs.seq_len,
            configs.n_filters,
            disable_moe=configs.disable_moe,
        )
        self.channel_mixer = torch.nn.Linear(n_dynamic_channels, configs.c_out)

        if self.disable_freq_enhance:
            self.freq_enhance.requires_grad_(False)

    # Same stages as FDN_PatchTST_RelPos6D.forward, applied only to the remaining modalities.
    # x_enc: (B, L, n * #modalities [+ n for q0]) from the input-ablation datasets.
    def forward(self, x_enc: torch.Tensor, channel_mask: torch.Tensor = None):
        batch_size, input_length, input_size = x_enc.size()
        if input_size != self.enc_in:
            raise ValueError(f"Expected {self.enc_in} input channels, got {input_size}.")

        if channel_mask is None:
            channel_mask = torch.ones(
                (batch_size, input_size), dtype=torch.bool, device=x_enc.device
            )

        # Zero out invalid channels and split time-varying inputs from q0
        mask_input = channel_mask.unsqueeze(1).expand(-1, input_length, -1)
        x_enc = x_enc.masked_fill(~mask_input, 0.0)

        n_dynamic_channels = self.n_joint * len(self.input_modalities)
        x_delta = x_enc[:, :, :n_dynamic_channels]
        if not self.exclude_jointpos:
            jointpos_0 = x_enc[:, :, n_dynamic_channels:]
        dynamic_mask = channel_mask[:, :n_dynamic_channels].unsqueeze(1)

        # RevIN and FEF over the remaining time-varying channels
        if self.use_rev_in:
            vars, means = torch.var_mean(x_delta, dim=1, keepdim=True)
            stdev = torch.sqrt(vars).clamp_min(1e-6)
            means = means.masked_fill(~dynamic_mask, 0.0)
            stdev = stdev.masked_fill(~dynamic_mask, 1.0)
            x_delta = (x_delta - means) / stdev

        if not self.disable_freq_enhance:
            x_delta = self.freq_enhance(x_delta)
        x_delta = x_delta.masked_fill(~dynamic_mask, 0.0)

        inputs = dict(
            zip(self.input_modalities, torch.split(x_delta, self.n_joint, dim=-1))
        )

        # Modality-specific encoders for the remaining modalities only
        enc_outputs = []
        attentions = []
        for modality in self.input_modalities:
            encoder = getattr(self, f"encoder_{modality}")
            enc_out, attention = encoder(inputs[modality])
            enc_outputs.append(enc_out)
            attentions.append(attention)

        enc_out = torch.cat(enc_outputs, dim=1)

        # Undo RevIN on the encoder outputs
        if self.use_rev_in:
            means = means.transpose(1, 2).unsqueeze(3)
            stdev = stdev.transpose(1, 2).unsqueeze(3)
            enc_out = enc_out * stdev + means

        # Add the q0 representation to the dq representation (only when positions are kept)
        if not self.exclude_jointpos:
            ic_jointpos = self.encoder_jointpos_0(
                jointpos_0[:, 0:1, :]
            ).unsqueeze(1)
            ic_jointpos = ic_jointpos.expand(-1, -1, self.num_patches, -1)
            enc_out = torch.cat(
                [
                    enc_out[:, : self.n_joint, ...] + ic_jointpos,
                    enc_out[:, self.n_joint :, ...],
                ],
                dim=1,
            )

        # Channel mixing, heads, FPF, and residual sampling as in FDN
        enc_out = enc_out.transpose(1, 3).contiguous()
        enc_out = self.channel_mixer(enc_out)
        enc_out = enc_out.transpose(1, 3).contiguous()

        trend, mu, logvar = self.decoder(enc_out)
        if not self.disable_freq_pass:
            trend = self.freq_pass_low(trend)
            mu = self.freq_pass_high(mu)
        res = self.sample_residual(mu, logvar)

        if self.output_attention:
            return (trend, res, mu, logvar), tuple(attentions)
        if self.disable_dethead:
            trend = torch.zeros_like(trend)
            return trend, res, mu, logvar
        if self.disable_probhead:
            res = torch.zeros_like(res)
            mu = torch.zeros_like(mu)
            logvar = torch.zeros_like(logvar)
        return trend, res, mu, logvar
