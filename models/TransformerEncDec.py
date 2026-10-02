import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    ConvLayer,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_wo_temporal
from layers.Filter import *
from layers.utils import *


"""
Baseline: vanilla Transformer encoder-decoder (O(L^2) attention) with non-autoregressive inference,
sequence-to-sequence estimator.

Estimates the wrench over the next T steps from the input history x'_{t-L+1:t} in one pass.
Adapted from https://github.com/thuml/iTransformer/blob/main/model/Transformer.py

References: A. Vaswani et al., Attention is all you need, NeurIPS, 2017.
H. Zhou et al., Informer, AAAI, 2021. https://doi.org/10.1609/aaai.v35i12.17325
"""

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.enc_in = 6 * 4
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding_wo_temporal(
            self.enc_in, configs.d_model, configs.dropout
        )
        self.dec_embedding = DataEmbedding_wo_temporal(
            self.enc_in, configs.d_model, configs.dropout
        )
        # Encoder
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
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(self, x_enc, mask=None):
        with torch.no_grad():
            # decoder input
            start_token = x_enc[:, -self.configs.label_len :, :]
            placeholder = torch.zeros_like(x_enc[:, : self.configs.pred_len, :])
            x_dec = torch.cat([start_token, placeholder], dim=1)

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)[
            :, -self.pred_len :, :
        ]  # [B, L, D]

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out
