import torch
import torch.nn as nn

import numpy as np
import random


def build_layers(layers_shapes, kernel_size=3, use_normalization=False):
    n = len(layers_shapes)
    assert n >= 2

    layers = []
    for i in range(n - 2):
        layers.append(nn.Conv1d(
            layers_shapes[i],
            layers_shapes[i + 1],
            kernel_size=(kernel_size,),
            padding=kernel_size // 2
        ))
        if use_normalization:
            layers.append(nn.InstanceNorm1d(layers_shapes[i + 1]))
        layers.append(nn.LeakyReLU(0.2))
    layers.append(nn.Conv1d(layers_shapes[-2], layers_shapes[-1], kernel_size=(1,)))

    return nn.Sequential(*layers)


class ConvModel(nn.Module):

    def __init__(
            self,
            out_channels=2,
            in_channels=1,
            hidden_channels_shapes=None,
            conv_kernel_size=3,
            use_normalization=False
    ):
        super().__init__()

        if hidden_channels_shapes is None:
            hidden_channels_shapes = []
        hidden_channels_shapes = [in_channels, *hidden_channels_shapes, out_channels]

        self.layers = build_layers(hidden_channels_shapes, conv_kernel_size, use_normalization)

    def forward(self, x):           # x: (batch, in_channels, seq_len)
        return self.layers(x)       # return: (batch, out_channels, seq_len)


def get_model(out_channels, in_channels=1, hidden_channels_shapes=None) -> ConvModel:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    np.random.seed(147)
    random.seed(0)
    return ConvModel(out_channels, in_channels, hidden_channels_shapes)
