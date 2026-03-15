import numpy as np
import matplotlib.pyplot as plt


def make_show_1_plot(ground_truth, noised, window=None, save=False):
    if window is None:
        window = [[-3, 3], [-3, 3]]
    fig, ax = plt.subplots()

    ax.grid(True, zorder=1, alpha=0.7)
    ax.scatter(ground_truth[0, 0], ground_truth[0, -1], color="green", s=40, zorder=3, label="Начальное состояние")

    ax.plot(ground_truth[:, 0], ground_truth[:, -1], color="black", zorder=2, label="Истинные значения (без шума)")
    ax.plot(noised[:, 0], noised[:, -1], color="red", linestyle='--', marker='o', markersize=5, zorder=2, label="Зашумлённые значения")

    # ax.set_title("Траектория линейной системы")

    ax.set_xlabel("$x_1$", fontsize=16)
    ax.set_ylabel(f"$x_{ground_truth.shape[1]}$", fontsize=16, rotation=0)
    ax.tick_params(labelsize=14)

    ax.set_xlim(window[0])
    ax.set_ylim(window[1])
    ax.legend(fontsize='x-small')

    plt.show()

    if save:
        fig.savefig("plot.pdf", dpi=300)


def make_show_1_plot_3d(ground_truth, noised, window=None, save=False):
    if window is None:
        window = [[-3, 3], [-3, 3], [-3, 3]]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.grid(True, zorder=1, alpha=0.7)
    ax.scatter(ground_truth[0, 0], ground_truth[0, 1], ground_truth[0, 2], color="green", s=40, zorder=3, label="Начальное состояние")

    ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], color="black", zorder=2, label="Истинные значения (без шума)")
    ax.plot(noised[:, 0], noised[:, 1], noised[:, 2], color="red", linestyle='--', marker='o', markersize=5, zorder=2, label="Зашумлённые значения")

    # ax.set_title("Траектория линейной системы")

    ax.set_xlabel("$x_1$", fontsize=16)
    ax.set_ylabel(f"$x_2$", fontsize=16, rotation=0)
    ax.set_zlabel(f"$x_3$", fontsize=16)
    ax.tick_params(labelsize=14)

    ax.set_xlim(window[0])
    ax.set_ylim(window[1])
    ax.set_zlim(window[2])
    ax.legend(fontsize='x-small')

    ax.view_init(elev=30, azim=135)

    plt.show()

    if save:
        fig.savefig("plot_3d.pdf", dpi=300)


def make_show_plot(ground_truth, noised, output, ground_truth_dot, dot_output, window=None, save=False):
    if window is None:
        window = [-3, 3]
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))

    ax[0].grid(True, zorder=1, alpha=0.7)
    ax[0].scatter(ground_truth[0, 0], ground_truth[0, 1], color="green", s=40, zorder=3, label="Начальное состояние")
    ax[1].grid(True, zorder=1, alpha=0.7)
    ax[1].scatter(ground_truth[0, 0], ground_truth[0, 1], color="green", s=40, zorder=3, label="Начальное состояние")
    ax[2].grid(True, zorder=1, alpha=0.7)
    ax[2].scatter(ground_truth_dot[0, 0], ground_truth_dot[0, 1], color="green", s=40, zorder=3, label="Начальное состояние")

    ax[0].plot(ground_truth[:, 0], ground_truth[:, 1], color="black", zorder=2, label="Истинные значения (без шума)")
    ax[0].plot(noised[:, 0], noised[:, 1], color="red", linestyle='--', marker='o', markersize=5, zorder=2, label="Зашумлённые значения")

    ax[1].plot(ground_truth[:, 0], ground_truth[:, 1], color="black", zorder=2, label="Истинные значения (без шума)")
    ax[1].plot(output[:, 0], output[:, 1], color="orange", linestyle='--', marker='o', markersize=5, zorder=2, label="Полученные значения (после модели)")

    ax[2].plot(ground_truth_dot[:, 0], ground_truth_dot[:, 1], color="black", zorder=2, label="Истинные значения производных (без шума)")
    ax[2].plot(dot_output[:, 0], dot_output[:, 1], color="blue", linestyle='--', marker='o', markersize=5, zorder=2, label="Полученные значения производных (после модели)")

    # ax.set_title("Траектория линейной системы")

    ax[0].set_xlabel("$x_1$", fontsize=16)
    ax[0].set_ylabel("$x_2$", fontsize=16, rotation=0)
    ax[0].tick_params(labelsize=14)

    ax[1].set_xlabel("$x_1$", fontsize=16)
    ax[1].set_ylabel("$x_2$", fontsize=16, rotation=0)
    ax[1].tick_params(labelsize=14)

    ax[2].set_xlabel("$x_1$", fontsize=16)
    ax[2].set_ylabel("$x_2$", fontsize=16, rotation=0)
    ax[2].tick_params(labelsize=14)

    ax[0].set_xlim(window)
    ax[0].set_ylim(window)
    ax[0].legend(fontsize='x-small')

    ax[1].set_xlim(window)
    ax[1].set_ylim(window)
    ax[1].legend(fontsize='x-small')

    ax[2].set_xlim([-10, 10])
    ax[2].set_ylim([-10, 10])
    ax[2].legend(fontsize='x-small', loc=3)

    plt.show()

    if save:
        fig.savefig("results.pdf", dpi=300)


def make_show_plot_3d(ground_truth, noised, output, ground_truth_dot, dot_output, window=None, save=False):
    if window is None:
        window = [-3, 3]
    fig = plt.figure(figsize=(21, 6))
    ax = [
        fig.add_subplot(1, 3, 1, projection='3d'),
        fig.add_subplot(1, 3, 2, projection='3d'),
        fig.add_subplot(1, 3, 3, projection='3d')
    ]

    ax[0].grid(True, zorder=1, alpha=0.7)
    ax[0].scatter(ground_truth[0, 0], ground_truth[0, 1], ground_truth[0, 2], color="green", s=40, zorder=3, label="Начальное состояние")
    ax[1].grid(True, zorder=1, alpha=0.7)
    ax[1].scatter(ground_truth[0, 0], ground_truth[0, 1], ground_truth[0, 2], color="green", s=40, zorder=3, label="Начальное состояние")
    ax[2].grid(True, zorder=1, alpha=0.7)
    ax[2].scatter(ground_truth_dot[0, 0], ground_truth_dot[0, 1], ground_truth_dot[0, 2], color="green", s=40, zorder=3, label="Начальное состояние")

    ax[0].plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], color="black", zorder=2, label="Истинные значения (без шума)")
    ax[0].plot(noised[:, 0], noised[:, 1], noised[:, 2], color="red", linestyle='--', marker='o', markersize=5, zorder=2, label="Зашумлённые значения")

    ax[1].plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], color="black", zorder=2, label="Истинные значения (без шума)")
    ax[1].plot(output[:, 0], output[:, 1], output[:, 2], color="orange", linestyle='--', marker='o', markersize=5, zorder=2, label="Полученные значения (после модели)")

    ax[2].plot(ground_truth_dot[:, 0], ground_truth_dot[:, 1], ground_truth_dot[:, 2], color="black", zorder=2, label="Истинные значения производных (без шума)")
    ax[2].plot(dot_output[:, 0], dot_output[:, 1], dot_output[:, 2], color="blue", linestyle='--', marker='o', markersize=5, zorder=2, label="Полученные значения производных (после модели)")

    # ax.set_title("Траектория линейной системы")

    ax[0].set_xlabel("$x_1$", fontsize=16)
    ax[0].set_ylabel("$x_2$", fontsize=16, rotation=0)
    ax[0].set_zlabel("$x_3$", fontsize=16, rotation=0)
    ax[0].tick_params(labelsize=14)

    ax[1].set_xlabel("$x_1$", fontsize=16)
    ax[1].set_ylabel("$x_2$", fontsize=16, rotation=0)
    ax[1].set_zlabel("$x_3$", fontsize=16, rotation=0)
    ax[1].tick_params(labelsize=14)

    ax[2].set_xlabel("$x_1$", fontsize=16)
    ax[2].set_ylabel("$x_2$", fontsize=16, rotation=0)
    ax[2].set_zlabel("$x_3$", fontsize=16, rotation=0)
    ax[2].tick_params(labelsize=14)

    ax[0].set_xlim(window[0])
    ax[0].set_ylim(window[1])
    ax[0].set_zlim(window[2])
    ax[0].legend(fontsize='x-small')
    ax[0].view_init(elev=30, azim=135)

    ax[1].set_xlim(window[0])
    ax[1].set_ylim(window[1])
    ax[1].set_zlim(window[2])
    ax[1].legend(fontsize='x-small')
    ax[1].view_init(elev=30, azim=135)

    ax[2].set_xlim([-200, 200])
    ax[2].set_ylim([-200, 200])
    ax[2].set_zlim([-200, 200])
    ax[2].legend(fontsize='x-small', loc=3)
    ax[2].view_init(elev=30, azim=135)

    plt.show()

    if save:
        fig.savefig("results_3d.pdf", dpi=300)


if __name__ == "__main__":

    # data = np.load("../datasets/linear_0_10_interval_100_points_-2_2_start_1e-2_noise.npz")
    # X, X_noised = data['X'], data["X_noised"]
    #
    # make_show_1_plot(
    #     X, X_noised,
    #     window=[[-3, 3], [-3, 3]]
    # )
    #
    # make_show_1_plot_3d(
    #     X, X_noised,
    #     window=[[-20, 20], [-20, 20], [0, 50]]
    # )

    X = np.load("../datasets/lorenz_0_10_interval_800_points_-8_7_27_start_1e-1_noise.npz")["X"]
    X_noised = np.load("../datasets/lorenz_0_10_interval_200_points_-8_7_27_start_1e-1_noise.npz")["X_noised"]

    make_show_1_plot_3d(
        X, X_noised,
        window=[[-20, 20], [-20, 20], [0, 50]]
    )