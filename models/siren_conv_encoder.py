import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from models.siren_nondefault_init import RktvModel
from models.conv import ConvModel

import numpy as np
import random


class RktvConvEncModel(nn.Module):

    def __init__(
            self,
            out_channels=2,
            embedding_size=4,

            siren_num_hidden_layers=3,
            siren_hidden_size=80,

            conv_hidden_layers=None,
            conv_out_channels=32,
            conv_kernel_size=3
    ):
        super().__init__()
        self.embedding_layer = gnn.TemporalEncoding(embedding_size)  # nn.Linear(1, embedding_size)
        self.encoder = ConvModel(
            conv_out_channels,
            out_channels,
            conv_hidden_layers,
            conv_kernel_size,
            use_normalization=True
        )
        self.decoder = RktvModel(
            out_channels,
            siren_num_hidden_layers,
            siren_hidden_size,
            embedding_size + conv_out_channels
        )

    def forward(self, time, states):
        # time: (seq_len,)
        # states: (out_channels, seq_len)

        x_embed = self.embedding_layer(time)                                    # (seq_len, embedding_size)
        x_encoder = self.encoder(states.unsqueeze(0)).squeeze().T               # (seq_len, conv_out_channels)

        x_encoder = torch.mean(x_encoder, dim=0).expand(x_embed.shape[0], -1)   # (seq_len, conv_out_channels)

        x_input = torch.cat([x_encoder, x_embed], dim=1)    # (seq_len, embedding_size + conv_out_channels)
        x_decoder = self.decoder(x_input)                   # (seq_len, out_channels)
        return x_decoder.T                                  # (out_channels, seq_len)


def get_model(
        out_channels=2,
        embedding_size=4,
        siren_num_hidden_layers=3,
        siren_hidden_size=80,
        conv_hidden_layers=None,
        conv_out_channels=32,
        conv_kernel_size=3
):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    np.random.seed(147)
    random.seed(0)
    return RktvConvEncModel(
        out_channels=out_channels,
        embedding_size=embedding_size,
        siren_num_hidden_layers=siren_num_hidden_layers,
        siren_hidden_size=siren_hidden_size,
        conv_hidden_layers=conv_hidden_layers,
        conv_out_channels=conv_out_channels,
        conv_kernel_size=conv_kernel_size
    )
