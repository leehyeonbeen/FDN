from layers.MLP import *
from layers.Patch import PatchTSTEncoder
from layers.LinearHead import FlattenLinearPatch
from layers.Filter import *
from layers.utils import *
from layers.LSTM import LayerNormLSTMCell
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.getcwd())


"""
Baseline: LSTM with layer normalization (an improved version of the cited original), sequence-to-point estimator.

Maps the input history x'_{t-L+1:t} = [q, qdot, qddot, u] to the current wrench W_t.

References: S. Kruzic, J. Music, I. Stancic, V. Papic, Neural network-based end-effector force
estimation for mobile manipulator on simulated uneven surfaces, SoftCOM, 2022.
https://doi.org/10.23919/SoftCOM55329.2022.9911383
S. Kruzic, J. Music, R. Kamnik, V. Papic, End-effector force and joint torque estimation of a 7-DoF
robotic manipulator using deep learning, Electronics, 2021. https://doi.org/10.3390/electronics10232963
"""

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = 6 * 4
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.lstm = LayerNormLSTMCell(self.enc_in, self.d_model)

        # e_layers controls the number of
        if configs.e_layers == 1:
            self.fc = nn.Linear(self.d_model, self.c_out)
        elif configs.e_layers > 1:
            self.fc = LayerNormMLP(
                self.d_model, self.c_out, self.d_model, configs.e_layers
            )
        else:
            raise ValueError("e_layers should be >= 1")

    def forward(self, x):
        batch_size, length, input_size = x.size()
        inputs = x.unbind(dim=1)

        hx = torch.zeros(batch_size, self.d_model, device=x.device)
        cx = torch.zeros(batch_size, self.d_model, device=x.device)
        state = (hx, cx)
        out_states = []
        for x_step in inputs:
            out, state = self.lstm(x_step, state)
            out_states += [out]
        out_states = torch.stack(out_states, dim=1)
        out = self.fc(out_states[:, -1, :])
        return out
