from siren import SIREN


def get_model(mlp_layers=None, in_features=1, out_features=2):
    if mlp_layers is None:
        mlp_layers = [80, 80, 80]

    model = SIREN(mlp_layers, in_features, out_features, 1.0, 30.0, initializer='siren', c=6)
    return model
