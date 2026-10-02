import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from layers.Filter import *
from layers.utils import *
from layers.Transformer_EncDec import *
from layers.SelfAttention_Family import *


"""
Baseline: iTransformer, sequence-to-sequence estimator.

Modified for our input setting with a channel-mixing projection to the 6 wrench channels.
Adapted from https://github.com/thuml/iTransformer/blob/main/model/iTransformer.py

Reference: Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, M. Long, iTransformer: Inverted transformers
are effective for time series forecasting, ICLR, 2024. https://arxiv.org/abs/2310.06625
"""

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = 24
        self.c_out = configs.c_out
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_rev_in = configs.use_rev_in
        self.disable_freq_enhance = configs.disable_freq_enhance
        self.disable_freq_pass = configs.disable_freq_pass

        self.linear = nn.Linear(configs.d_model, configs.pred_len)
        self.channel_mixer = nn.Linear(self.enc_in, configs.c_out)

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            dropout=configs.dropout,
        )
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

    def forward(self, x_enc):
        if self.use_rev_in:
            means = x_enc.mean(dim=1, keepdim=True)
            stdev = torch.std(x_enc, dim=1, keepdim=True) + 1e-6
            x_enc = (x_enc - means) / stdev

        batch_size, seq_len, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(
            x_enc
        )  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        if self.use_rev_in:
            enc_out = enc_out.transpose(1, 2) * stdev + means
            enc_out = enc_out.transpose(1, 2)

        # channel projection
        enc_out = self.channel_mixer(enc_out.transpose(1, 2)).transpose(1, 2)

        # B N E -> B N S -> B S N
        dec_out = self.linear(enc_out).permute(0, 2, 1)[
            :, :, :N
        ]  # filter the covariates

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out
