import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Baseline: CNN, sequence-to-point estimator.

Implements the CNN part of the original STFFM architecture (channel convolution and pooling in
Fig. 6 and Table 3 of the reference) and maps the input history x'_{t-L+1:t} = [q, qdot, qddot, u]
to the current wrench W_t.

Reference: M.-Z. Pan, J.-A. Li, Z. Li, K. Liang, T.-C. Su, K. Liang, G.-B. Bian, A graph robot network
for force observer of teleoperation systems, IEEE/ASME TMECH, 2025. https://doi.org/10.1109/TMECH.2024.3402995
"""

class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.enc_in = 6 * 4  # 6DoF representations [q, dq, ddq, tau]
        self.seq_len = configs.seq_len
        self.c_out = configs.c_out  # 6F/T

        self.conv1 = nn.Conv2d(1, configs.d_model, kernel_size=(7, 1), padding=(3, 0))
        self.norm1 = nn.BatchNorm2d(configs.d_model)
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 1), stride=(1, 1))

        self.conv2 = nn.Conv2d(
            configs.d_model, configs.d_model // 2, kernel_size=(7, 1), padding=(3, 0)
        )
        self.norm2 = nn.BatchNorm2d(configs.d_model // 2)

        self.conv3 = nn.Conv2d(
            configs.d_model // 2,
            configs.d_model // 4,
            kernel_size=(9, 1),
            padding=(4, 0),
        )
        self.norm3 = nn.BatchNorm2d(configs.d_model // 4)

        # Original paper does not specify the head architecture.
        self.fc = nn.Linear(
            (configs.d_model // 4) * (self.enc_in - 1), configs.c_out
        )  # time pooling -> fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, input_size = x.size()
        out = x.transpose(1, 2).unsqueeze(1)  # B,1,Cin,L
        out = self.conv1(out)  # B,D,Cin,L
        out = self.norm1(out)
        out = F.relu(out)
        out = self.pool1(out)  # B,D,Cin-1,L

        out = self.conv2(out)  # B,D/2,Cin-1,L
        out = self.norm2(out)
        out = F.relu(out)

        out = self.conv3(out)  # B,D/4,Cin-1,L
        out = self.norm3(out)
        out = F.relu(out)

        out = out.mean(dim=-1)  # time pooling (B,D/4,Cin-1)
        out = out.flatten(1)
        out = self.fc(out)
        return out
