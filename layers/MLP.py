import torch
import torch.nn as nn


# MLP with GELU activations
class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_hl: int = 1,
        dropout: float = 0,
    ) -> None:
        super().__init__()

        self.num_hidden_layers = num_hl
        self.mlp = []
        self.mlp.append(nn.Linear(input_size, hidden_size))
        self.mlp.append(nn.GELU())
        self.mlp.append(nn.Dropout(dropout))
        for i in range(num_hl - 1):
            self.mlp.append(nn.Linear(hidden_size, hidden_size))
            self.mlp.append(nn.GELU())
            self.mlp.append(nn.Dropout(dropout))
        self.mlp.append(nn.Linear(hidden_size, output_size))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        return out


# MLP with LayerNorm and GELU after each of the num_hl hidden layers
class LayerNormMLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_hl: int = 1,
        dropout: float = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert num_hl >= 1, "num_hl must be at least 1"
        self.num_hl = num_hl
        self.linear_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(hidden_size, output_size, bias=bias)
        for i in range(num_hl):
            if i == 0:
                self.linear_layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.linear_layers.append(nn.Linear(hidden_size, hidden_size))
            self.norm_layers.append(nn.LayerNorm(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_hl):
            x = self.linear_layers[i](x)
            x = self.norm_layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        out = self.fc_out(x)
        return out
