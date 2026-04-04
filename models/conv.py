import torch
import torch.nn as nn

import numpy as np
import random


def build_layers(layers_shapes, kernel_size=3):
    n = len(layers_shapes)
    assert n >= 2

    layers = []
    for i in range(n - 2):
        layers.append(nn.Conv1d(layers_shapes[i], layers_shapes[i + 1], kernel_size=kernel_size, padding=kernel_size // 2))
        layers.append(nn.ReLU())
    layers.append(nn.Conv1d(layers_shapes[-2], layers_shapes[-1], kernel_size=1))

    return nn.Sequential(*layers)


class ConvModel(nn.Module):

    def __init__(self, out_channels=2, in_channels=1, hidden_channels_shapes=None, conv_kernel_size=3):
        super().__init__()

        if hidden_channels_shapes is None:
            hidden_channels_shapes = []
        hidden_channels_shapes = [in_channels, *hidden_channels_shapes, out_channels]

        self.layers = build_layers(hidden_channels_shapes, conv_kernel_size)

    def forward(self, x):
        # x: (batch, 1, seq_len)
        return self.layers(x)


def get_model(out_channels, in_channels=1, hidden_channels_shapes=None) -> ConvModel:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    np.random.seed(147)
    random.seed(0)
    return ConvModel(out_channels, in_channels, hidden_channels_shapes)
