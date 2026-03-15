from models.utils import infer_model
from data.utils.loader import get_data
from data.utils.visualize import make_show_plot

data_path = '../data/datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz'
timestamps, data, data_derivatives, data_noised = get_data(data_path)

model_path = "../models_checkpoints/model_9/1600_epoch/model.pth"
output, output_derivatives = infer_model(model_path, timestamps)

data_better_path = '../data/datasets/linear_0_10_interval_400_points_-2_2_start_1e-1_noise.npz'
timestamps__better, data__better, data_derivatives__better, data_noised__better = get_data(data_better_path)

# make_show_plot(data, data_noised, output, data_derivatives, output_derivatives)
make_show_plot(data__better, data_noised, output, data_derivatives__better, output_derivatives)

