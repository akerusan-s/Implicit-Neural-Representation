from configs import *
from training.train import train_model
from models.siren_nondefault_init import get_model

config = Config22()
model = train_model(config, get_model(config.out_dim, config.num_hidden_layers, config.hidden_size))
