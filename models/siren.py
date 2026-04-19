import random

import torch
import torch.nn as nn

import numpy as np


class RktvModel(nn.Module):

    def __init__(
            self,
            output_dim,
            num_hidden_layers=3,
            hidden_size=80,
            input_dim=1
    ):

        super(RktvModel, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_size)

        ###
        torch.nn.init.uniform_(self.input_layer.weight, -30 * (6 / input_dim) ** 0.5, 30 * (6 / input_dim) ** 0.5)
        torch.nn.init.zeros_(self.input_layer.bias)
        ###

        self.hidden_layers = nn.Sequential()
        for _ in range(num_hidden_layers):
            hidden_linear_layer = nn.Linear(hidden_size, hidden_size)

            ###
            torch.nn.init.uniform_(hidden_linear_layer.weight, -1 * (6 / hidden_size) ** 0.5, (6 / hidden_size) ** 0.5)
            torch.nn.init.zeros_(hidden_linear_layer.bias)
            ###

            self.hidden_layers.append(hidden_linear_layer)

        self.output_layer = nn.Linear(hidden_size, output_dim)

        ###
        torch.nn.init.uniform_(self.output_layer.weight, -1 * (6 / hidden_size) ** 0.5, (6 / hidden_size) ** 0.5)
        torch.nn.init.zeros_(self.output_layer.bias)
        ###

    def forward(self, t):
        t = torch.sin(self.input_layer(t))
        for layer in self.hidden_layers:
            t = torch.sin(layer(t))
        return self.output_layer(t)


def get_model(
        output_dim,
        num_hidden_layers=3,
        hidden_size=80,
        input_dim=1
) -> RktvModel:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    np.random.seed(147)
    random.seed(0)

    return RktvModel(
        output_dim,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        input_dim=input_dim
    )
