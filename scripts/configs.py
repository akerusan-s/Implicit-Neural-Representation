class DefaultConfig:
    epochs = 100
    lr = 5e-4
    out_dim = 2
    num_hidden_layers = 3
    hidden_size = 80
    c_hyperparam = [1, 1, 1e-5]
    log_interval = 10
    save_interval = 50
    save = True
    save_path = '../models_checkpoints/model'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'


class Config1:
    epochs = 100
    lr = 5e-4
    out_dim = 2
    num_hidden_layers = 3
    hidden_size = 80
    c_hyperparam = [1, 1, 1e-5]
    log_interval = 1
    save_interval = 50
    save = True
    save_path = '../models_checkpoints/model_1'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'


class Config2:
    epochs = 2000
    lr = 5e-4
    out_dim = 2
    num_hidden_layers = 3
    hidden_size = 80
    c_hyperparam = [1, 1, 1e-5]
    log_interval = 10
    save_interval = 500
    save = True
    save_path = '../models_checkpoints/model_2'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'


class Config3:
    epochs = 500
    lr = 5e-4
    out_dim = 2
    num_hidden_layers = 3
    hidden_size = 80
    c_hyperparam = [1, 1, 1e-2]
    log_interval = 10
    save_interval = 100
    save = True
    save_path = '../models_checkpoints/model_3'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'


class Config4(DefaultConfig):
    epochs = 500
    save_interval = 250
    c_hyperparam = [1, 1, 1e-3]
    save_path = '../models_checkpoints/model_4'


class Config5(DefaultConfig):
    epochs = 1000
    save_interval = 500
    c_hyperparam = [1, 1, 1e-3]
    save_path = '../models_checkpoints/model_5'


class Config6:
    model_type = 'nondefault_init'
    epochs = 600
    lr = 5e-4
    out_dim = 2
    num_hidden_layers = 3
    hidden_size = 80
    c_hyperparam = [1, 1, 1e-3]
    log_interval = 10
    save_interval = 200
    save = True
    save_path = '../models_checkpoints/model_6'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'


class Config7:
    model_type = 'nondefault_init'
    epochs = 1000
    lr = 5e-4
    out_dim = 2
    num_hidden_layers = 3
    hidden_size = 80
    c_hyperparam = [1, 1, 1e-2]
    log_interval = 10
    save_interval = 200
    save = True
    save_path = '../models_checkpoints/model_7'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'


class Config8:
    model_type = 'nondefault_init'
    epochs = 1000
    lr = 5e-4
    out_dim = 2
    num_hidden_layers = 3
    hidden_size = 80
    c_hyperparam = [1, 1, 1e-4]
    log_interval = 10
    save_interval = 200
    save = True
    save_path = '../models_checkpoints/model_8'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'


class Config9:
    model_type = 'nondefault_init'
    epochs = 2000
    lr = 5e-4
    out_dim = 2
    num_hidden_layers = 3
    hidden_size = 80
    c_hyperparam = [1, 1, 1e-4]
    log_interval = 10
    save_interval = 400
    save = True
    save_path = '../models_checkpoints/model_9'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'


class Config10:
    model_type = 'conv_rktv'
    device = 'cuda'
    epochs = 1000
    lr = 5e-4
    out_dim = 2
    c_hyperparam = [1, 1, 1e-4]
    log_interval = 10
    save_interval = 400
    save = True
    save_path = '../models_checkpoints/model_10'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'
    in_channels = 1,
    siren_num_hidden_layers = 3
    siren_hidden_size = 80
    conv_hidden_layers = [16, 32, 64, 32, 16]
    conv_out_channels = 2


class Config11:
    model_type = 'conv_rktv'
    device = 'cuda'
    epochs = 1000
    lr = 5e-4
    c_hyperparam = [1, 1, 1e-4]
    log_interval = 10
    save_interval = 200
    save = True
    save_path = '../models_checkpoints/model_11'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'
    out_dim = 2
    in_channels = 1,
    siren_num_hidden_layers = 3
    siren_hidden_size = 40
    conv_hidden_layers = [16, 16]
    conv_out_channels = 2


class Config12:
    model_type = 'conv_rktv'
    device = 'cuda'
    epochs = 1000
    lr = 3e-4
    c_hyperparam = [1, 1, 1e-4]
    log_interval = 10
    save_interval = 200
    save = True
    save_path = '../models_checkpoints/model_12'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'
    out_dim = 2
    in_channels = 1,
    siren_num_hidden_layers = 3
    siren_hidden_size = 40
    conv_hidden_layers = [16]
    conv_out_channels = 16


class Config13:
    model_type = 'conv_rktv'
    device = 'cuda'
    epochs = 2000
    lr = 3e-4
    c_hyperparam = [1, 1, 1e-4]
    log_interval = 10
    save_interval = 200
    save = True
    save_path = '../models_checkpoints/model_13'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'
    out_dim = 2
    in_channels = 1,
    siren_num_hidden_layers = 3
    siren_hidden_size = 60
    conv_hidden_layers = [16]
    conv_out_channels = 16
    conv_kernel_size = 5


class Config14:
    model_type = 'conv_rktv'
    device = 'cuda'
    epochs = 2000
    lr = 3e-4
    c_hyperparam = [1, 1, 1e-4]
    log_interval = 10
    save_interval = 200
    save = True
    save_path = '../models_checkpoints/model_14'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'
    out_dim = 2
    in_channels = 1,
    siren_num_hidden_layers = 3
    siren_hidden_size = 80
    conv_hidden_layers = [16, 16]
    conv_out_channels = 16
    conv_kernel_size = 7


class Config15:
    model_type = 'conv_siren_rktv_residual'
    device = 'cpu'
    epochs = 3000
    lr = 5e-4
    c_hyperparam = [1, 1, 1e-5]
    log_interval = 1
    save_interval = 500
    save = True
    save_path = '../models_checkpoints/model_15'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'
    out_dim = 2
    in_channels = 1
    siren_num_hidden_layers = 3
    siren_hidden_size = 80
    conv_hidden_layers = [4]
    model_dim = 2
    conv_kernel_size = 5


class Config16:
    model_type = 'conv_siren_rktv_encoder'
    device = 'cuda'
    epochs = 3000
    lr = 5e-4
    c_hyperparam = [1, 1, 1e-3]
    log_interval = 1
    save_interval = 1000
    save = True
    save_path = '../models_checkpoints/model_16'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'
    out_channels = 2
    siren_num_hidden_layers = 3
    siren_hidden_size = 80
    conv_hidden_layers = [4, 8]
    conv_kernel_size = 5
    conv_out_channels = 16
    embedding_size = 2


class Config17:
    model_type = 'conv_siren_rktv_encoder'
    device = 'cuda'
    epochs = 1000
    lr = 5e-4
    c_hyperparam = [1, 1, 1e-3]
    log_interval = 1
    save_interval = 500
    save = True
    save_path = '../models_checkpoints/model_17'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'
    out_channels = 2
    siren_num_hidden_layers = 3
    siren_hidden_size = 80
    conv_hidden_layers = [4, 8]
    conv_kernel_size = 5
    conv_out_channels = 16
    embedding_size = 16


class Config18:
    model_type = 'conv_siren_rktv_'
    device = 'cpu'
    epochs = 5000
    lr = 5e-4
    c_hyperparam = [1, 1, 1e-4]
    log_interval = 100
    save_interval = 2500
    save = True
    save_path = '../models_checkpoints/model_18'
    data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'
    out_channels = 2
    siren_num_hidden_layers = 2
    siren_hidden_size = 40
    conv_hidden_layers = [16, 32]
    conv_kernel_size = 5
    conv_out_channels = 32
    embedding_size = 8


class Config19:
    model_type = 'conv_siren_rktv_residual'
    device = 'cpu'
    epochs = 7500
    lr = 1e-4
    c_hyperparam = [1, 1, 1e-5]
    log_interval = 50
    save_interval = 2500
    save = True
    save_path = '../models_checkpoints/model_19'
    data_path = '../data/datasets/lorenz_0_10_interval_200_points_-8_7_27_start_1e-1_noise.npz'
    out_dim = 3
    in_channels = 1
    siren_num_hidden_layers = 3
    siren_hidden_size = 80
    conv_hidden_layers = [4, 8]
    model_dim = 16
    conv_kernel_size = 5


class Config20:
    model_type = 'conv_siren_rktv_encoder'
    device = 'cpu'
    epochs = 7500
    lr = 5e-4
    c_hyperparam = [1, 1, 1e-6]
    log_interval = 100
    save_interval = 7500
    save = True
    save_path = 'model_20'
    data_path = 'lorenz_0_10_interval_200_points_-8_7_27_start_1e-1_noise.npz'
    embedding_size = 32
    out_channels = 3
    siren_num_hidden_layers = 3
    siren_hidden_size = 80
    conv_hidden_layers = [8, 16, 32, 64]
    conv_out_channels = 64
    conv_kernel_size = 5
    use_normalization = False
