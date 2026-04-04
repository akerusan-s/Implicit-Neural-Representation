from configs import *
from training.train_siren_conv import train_model
from models.siren_conv import get_model

config = Config13()
model = get_model(
    config.out_dim,
    config.in_channels,
    config.siren_num_hidden_layers,
    config.siren_hidden_size,
    config.conv_hidden_layers,
    config.conv_out_channels,
    config.conv_kernel_size
)
train_model(config, model)
