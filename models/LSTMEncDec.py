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
Baseline: autoregressive LSTM encoder-decoder (LSTM-ED) with layer normalization (an improved version of the cited original),
sequence-to-sequence estimator.

Encodes the input history x'_{t-L+1:t} and decodes the wrench over the next T steps one step at a time.

References: I.-F. Kao, Y. Zhou, L.-C. Chang, F.-J. Chang, Exploring a long short-term memory based
encoder-decoder framework for multi-step-ahead flood forecasting, J. Hydrol., 2020.
https://doi.org/10.1016/j.jhydrol.2020.124631
S. Kruzic, J. Music, R. Kamnik, V. Papic, End-effector force and joint torque estimation of a 7-DoF
robotic manipulator using deep learning, Electronics, 2021. https://doi.org/10.3390/electronics10232963
"""

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = 6 * 4
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.lstm_enc = LayerNormLSTMCell(self.enc_in, self.d_model)
        self.lstm_dec = LayerNormLSTMCell(self.c_out, self.d_model)

        # e_layers sets the output head: a linear layer if 1, otherwise an MLP with e_layers hidden layers
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
        outputs = []

        # encoder loop
        for x_step in inputs:
            out, state = self.lstm_enc(x_step, state)
        # decoder loop
        dec_y = torch.zeros(batch_size, self.c_out, device=x.device)
        for t in range(self.pred_len):
            out, state = self.lstm_dec(dec_y, state)
            dec_y = self.fc(out)
            outputs += [dec_y]
        outputs = torch.stack(outputs, dim=1)
        return outputs
