from layers.Embed import TokenEmbedding, PositionalEmbedding
from layers.utils import *
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention
import math
import os
import sys

sys.path.append(os.getcwd())


# PatchTST encoder: channel-independent patching, patch and positional embeddings, Transformer encoder.
# (B, L, C) -> (B, C, N, D), with N = (L - P) // P + 2 patches of length P.
class PatchTSTEncoder(nn.Module):
    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        d_model: int,
        d_ff: int,
        e_layers: int,
        dropout: float,
        n_heads: int,
        patch_len: int = 8,
        activation: str = "gelu",
        output_attention: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len

        self.patch_stride = patch_len
        self.num_patches = (seq_len - patch_len) // self.patch_stride + 2

        self.embedding_value = TokenEmbedding(patch_len, d_model)
        self.embedding_pos = PositionalEmbedding(d_model, self.num_patches)

        self.dropout = nn.Dropout(dropout)

        # self.rev_in_norm = ReversibleInstanceNorm(enc_in, affine=False)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

    def patch_series(self, x):
        # Repeat the last step P times so that the last patch is complete
        pad = x[:, -1:, :].expand(-1, self.patch_stride, -1)
        # (B, L + P, C)
        x = torch.cat([x, pad], dim=1)
        # Channel-independent patching
        x = x.permute(0, 2, 1).flatten(0, 1)  # (B*C, L + P)
        patches = x.unfold(1, self.patch_len, self.patch_stride)  # (B*C, N, P)
        return patches

    def forward(self, x: torch.Tensor):
        batch_size, input_length, input_size = x.size()

        # RevIN is applied in the models, not here
        # x = self.rev_in_norm(x, mode="norm")  # (B,L,Cin)
        # patching
        patches = self.patch_series(x)
        # Patch embedding + positional embedding
        enc_out = self.dropout(
            self.embedding_value(patches) + self.embedding_pos(patches)
        )  # (B*C, N, D)
        # Transformer encoder
        enc_out, attns = self.encoder(enc_out)  # (B*C, N, D)
        # Back to per-channel layout
        enc_out = enc_out.reshape(batch_size, input_size, -1)  # (B,C,N*D)
        # enc_out = self.rev_in_norm(enc_out.transpose(1, 2), mode="denorm").transpose(
        #     1, 2
        # )  # (B,C,N*D)
        # (B, C, N, D)
        enc_out = enc_out.reshape(
            batch_size, input_size, self.num_patches, self.d_model
        )
        return enc_out, attns  # (B,C,N,D)


# Original RevIN (unused: the models apply a modified RevIN to the encoder outputs)
class ReversibleInstanceNorm(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True)
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + self.eps)

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev + self.mean
        return x
