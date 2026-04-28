import numpy as np

from scipy.signal import savgol_filter

from scipy.interpolate import make_smoothing_spline


class SavGolFilter:

    def __init__(self, window_size, poly_order):
        self.window_size = window_size
        self.poly_order = poly_order

    def apply(self, data, time_step=0.1, axis=0):
        states = savgol_filter(data, self.window_size, self.poly_order, axis=axis)
        derivatives = savgol_filter(data, self.window_size, self.poly_order, deriv=1, axis=axis, delta=time_step)
        return states, derivatives


class Spline:

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def apply(self, data, time):
        states = []
        derivatives = []
        for i in range(data.shape[1]):
            spl = make_smoothing_spline(time, data[:, i], lam=self.lambda_)
            states.append(spl(time))
            derivatives.append(spl.derivative()(time))
        return np.array(states).T, np.array(derivatives).T
