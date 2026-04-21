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
            conv_kernel_size=3,
            model_dim=4
    ):
        super().__init__()
        self.embedding_layer = nn.Linear(in_channels, model_dim)
        self.siren_model = RktvModel(model_dim, siren_num_hidden_layers, siren_hidden_size, model_dim)
        self.conv_model = ConvModel(model_dim, model_dim, conv_hidden_layers, conv_kernel_size)
        self.linear_head = nn.Linear(model_dim, out_channels)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        x = torch.transpose(self.embedding_layer(torch.transpose(x, 1, 2)), 1, 2)  # x: (batch, model_dim, seq_len)
        x = x + self.conv_model(x)                                                 # x: (batch, model_dim, seq_len)
        x = x + torch.transpose(self.siren_model(torch.transpose(x, 1, 2)), 1, 2)
        x = torch.transpose(self.linear_head(torch.transpose(x, 1, 2)), 1, 2)
        return x


def get_model(
        out_channels=2,
        in_channels=1,
        siren_num_hidden_layers=3,
        siren_hidden_size=80,
        conv_hidden_layers=None,
        conv_kernel_size=3,
        model_dim=4
):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    np.random.seed(147)
    random.seed(0)
    return RktvConvModel(
        out_channels=out_channels,
        in_channels=in_channels,
        siren_num_hidden_layers=siren_num_hidden_layers,
        siren_hidden_size=siren_hidden_size,
        conv_hidden_layers=conv_hidden_layers,
        conv_kernel_size=conv_kernel_size,
        model_dim=model_dim
    )
