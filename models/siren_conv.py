import torch
import torch.nn as nn

from models.siren_nondefault_init import RktvModel
from models.conv import ConvModel

import numpy as np
import random


class RktvConvModel(nn.Module):

    def __init__(
            self,
            out_channels=2,
            in_channels=1,
            siren_num_hidden_layers=3,
            siren_hidden_size=80,
            conv_hidden_layers=None,
            conv_out_channels=2,
            conv_kernel_size=3
    ):
        super().__init__()
        self.siren_model = RktvModel(out_channels, siren_num_hidden_layers, siren_hidden_size, conv_out_channels)
        self.conv_model = ConvModel(conv_out_channels, in_channels, conv_hidden_layers, conv_kernel_size)

    def forward(self, x):
        # x: (batch, 1, seq_len)
        x = self.conv_model(x)
        x = self.siren_model(x.squeeze().T).T.unsqueeze(0)
        return x


def get_model(
        out_channels=2,
        in_channels=1,
        siren_num_hidden_layers=3,
        siren_hidden_size=80,
        conv_hidden_layers=None,
        conv_out_channels=2,
        conv_kernel_size=3
):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    np.random.seed(147)
    random.seed(0)
    return RktvConvModel(
        out_channels,
        in_channels,
        siren_num_hidden_layers,
        siren_hidden_size,
        conv_hidden_layers,
        conv_out_channels,
        conv_kernel_size
    )
