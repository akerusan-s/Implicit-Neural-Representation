import random
import numpy as np
from scipy.integrate import solve_ivp

from systems.linear import linear_system
from systems.lorenz import lorenz_system

from pathlib import Path


def model_system_X(system, t_eval, zero_state):
    sol = solve_ivp(
        system,
        y0=zero_state,
        t_span=[t_eval[0], t_eval[-1]],
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-10,
        method="RK45",
    )
    return sol.y.T


def model_system_X_dot(system, X_values):
    res = []
    for i in X_values:
        res.append(system(0, i))
    return np.array(res)


def model_system(system, t_eval, zero_state):
    X_values = model_system_X(system, t_eval, zero_state)
    X_dot = model_system_X_dot(system, X_values)
    return (t_eval, X_values, X_dot)


def get_noised_X(X_values, sigma=1e-1):
    n_dim = X_values.shape[1]
    L = np.sqrt(np.sum(np.var(X_values, axis=0)) / n_dim)
    variance = sigma * L

    noise = np.random.normal(0, variance, size=X_values.shape)
    X_noised = X_values + noise

    return X_noised


if __name__ == "__main__":
    np.random.seed(147)
    random.seed(0)
    #
    # ts, X, X_dot = model_system(
    #     linear_system,
    #     np.linspace(0, 10, 100),
    #     [-2, 2]
    # )
    # X_noised = get_noised_X(X, 0.05)
    #
    # np.savez('../datasets/linear_0_10_interval_100_points_-2_2_start_5e-2_noise.npz',
    #          ts=ts, X=X, X_dot=X_dot, X_noised=X_noised)
    #
    # ts, X, X_dot = model_system(
    #     linear_system,
    #     np.linspace(0, 10, 100),
    #     [-2, 2]
    # )
    # X_noised = get_noised_X(X, 1e-2)
    #
    # np.savez('../datasets/linear_0_10_interval_100_points_-2_2_start_1e-2_noise.npz',
    #          ts=ts, X=X, X_dot=X_dot, X_noised=X_noised)
    #
    # ts, X, X_dot = model_system(
    #     linear_system,
    #     np.linspace(0, 10, 1000),
    #     [-2, 2]
    # )
    # X_noised = get_noised_X(X)
    #
    # np.savez('../datasets/linear_0_10_interval_1000_points_-2_2_start_1e-1_noise.npz',
    #          ts=ts, X=X, X_dot=X_dot, X_noised=X_noised)
    #
    # ts, X, X_dot = model_system(
    #     linear_system,
    #     np.linspace(0, 10, 1000),
    #     [-2, 2]
    # )
    # X_noised = get_noised_X(X, 1e-2)
    #
    # np.savez('../datasets/linear_0_10_interval_1000_points_-2_2_start_1e-2_noise.npz',
    #          ts=ts, X=X, X_dot=X_dot, X_noised=X_noised)

    # ts, X, X_dot = model_system(
    #     lorenz_system,
    #     np.linspace(0, 10, 200),
    #     [-8, 7, 27]
    # )
    # X_noised = get_noised_X(X)
    #
    # np.savez('../datasets/lorenz_0_10_interval_200_points_-8_7_27_start_1e-1_noise.npz',
    #          ts=ts, X=X, X_dot=X_dot, X_noised=X_noised)

    ts, X, X_dot = model_system(
        lorenz_system,
        np.linspace(0, 10, 200),
        [-8, 7, 27]
    )
    X_noised = get_noised_X(X, 5e-2)

    np.savez('../datasets/lorenz_0_10_interval_200_points_-8_7_27_start_5e-2_noise.npz',
             ts=ts, X=X, X_dot=X_dot, X_noised=X_noised)

    # m = X.shape[0]
    # data = np.load('../datasets/linear_0_10_interval_100_points_-2_2_start_1e-1_noise.npz')
    # print(data["a"])
