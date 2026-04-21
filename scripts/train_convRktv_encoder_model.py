from configs import *
from training.train_siren_conv_encoder import train_model
from models.siren_conv_encoder import get_model

config = Config18()
model = get_model(
    out_channels=config.out_channels,
    embedding_size=config.embedding_size,
    siren_num_hidden_layers=config.siren_num_hidden_layers,
    siren_hidden_size=config.siren_hidden_size,
    conv_hidden_layers=config.conv_hidden_layers,
    conv_out_channels=config.conv_out_channels,
    conv_kernel_size=config.conv_kernel_size
)
train_model(config, model)
