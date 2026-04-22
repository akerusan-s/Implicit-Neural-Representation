import optuna
import torch

from configs import *
from data.utils.loader import get_data
from models.siren_conv_encoder import get_model
from training.train_siren_conv_encoder import train


def train_model(config):

    def objective(trial):

        config.lr = trial.suggest_float("lr", 1e-5, 1e-2)
        config.weight_decay = trial.suggest_float("weight_decay", 0, 1e-1)
        config.c_hyperparam = [1, 1, trial.suggest_float("c3", 0, 1e-2)]

        config.siren_num_hidden_layers = trial.suggest_int("siren_num_layers", 1, 5)
        config.siren_hidden_size = trial.suggest_int("siren_layer_size", 8, 256)

        config.conv_hidden_layers = [trial.suggest_int(f"conv_layer_size_{i}", 2, 128) for i in range(trial.suggest_int("conv_num_layers", 1, 5))]
        config.conv_kernel_size = trial.suggest_int("conv_kernel_size", 3, 27)
        config.conv_out_channels = trial.suggest_int("conv_out_layer_size", 2, 128)

        config.embedding_size = trial.suggest_int("embedding_size", 2, 128)

        print()
        print("Current params")
        for key_, value_ in trial.params.items():
            print("    {}: {}".format(key_, value_))

        model = get_model(
            out_channels=config.out_channels,
            embedding_size=config.embedding_size,
            siren_num_hidden_layers=config.siren_num_hidden_layers,
            siren_hidden_size=config.siren_hidden_size,
            conv_hidden_layers=config.conv_hidden_layers,
            conv_out_channels=config.conv_out_channels,
            conv_kernel_size=config.conv_kernel_size
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        timestamps, data, data_derivatives, data_noised = get_data(config.data_path)

        return train(
            model,
            optimizer,
            config,
            timestamps,
            data,
            data_derivatives,
            data_noised,
            return_metric=True
        )

    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name="conv_siren_rktv_encoder",
        direction='minimize'
    )
    study.optimize(objective, n_trials=100)

    print()
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))


train_config = Config20()
train_model(train_config)
