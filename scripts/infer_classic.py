from data.utils.loader import get_data
from metrics.metrics import metric
from models.classic_models import Spline, SavGolFilter


data_path = '../data/datasets/lorenz_0_10_interval_200_points_-8_7_27_start_1e-1_noise.npz'
timestamps, data, data_derivatives, data_noised = get_data(data_path)

savgol = SavGolFilter(9, 6)
savgol_states, savgol_derivatives = savgol.apply(data_noised, 1 / 20)
print("Sav Gol Filter")
print(f"States: {metric(data, savgol_states)}")                             # 0.04136292263865471
print(f"Derivatives: {metric(data_derivatives, savgol_derivatives)}")       # 0.23040726780891418
print()

spline = Spline(7e-5)
spline_states, spline_derivatives = spline.apply(data_noised, timestamps)
print("Spline")
print(f"States: {metric(data, spline_states)}")                             # 0.04824995994567871
print(f"Derivatives: {metric(data_derivatives, spline_derivatives)}")       # 0.23752613365650177
print()
