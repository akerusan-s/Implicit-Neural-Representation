from configs import *
from training.train_siren_conv import train_model
from models.siren_conv import get_model

config = Config19()
model = get_model(
    out_channels=config.out_dim,
    in_channels=config.in_channels,
    siren_num_hidden_layers=config.siren_num_hidden_layers,
    siren_hidden_size=config.siren_hidden_size,
    conv_hidden_layers=config.conv_hidden_layers,
    conv_kernel_size=config.conv_kernel_size,
    model_dim=config.model_dim
)
train_model(config, model)
