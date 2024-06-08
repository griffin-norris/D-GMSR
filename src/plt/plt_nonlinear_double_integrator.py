import numpy as np
import typing as T
import matplotlib.pyplot as plt


def ndi_plot(results, params, save_fig, fig_format, fig_png_dpi):
    fig, axs = plt.subplots(2, 2, gridspec_kw={"height_ratios": [5, 3]}, figsize=(9, 6))

    gs = axs[0, 0].get_gridspec()
    for ax in axs[0, :]:
        ax.remove()
    axa = fig.add_subplot(gs[0, :])

    circle = plt.Circle(
        params["p_w"][:2], params["p_w"][2], color="blue", clip_on=False, fill=False
    )
    axa.add_patch(circle)

    axa.plot(
        params["X_last"][0, :],
        params["X_last"][1, :],
        c="black",
        label="Initial Trajectory",
    )
    axa.scatter(params["X_last"][0, :], params["X_last"][1, :], c="black")

    axa.plot(
        results["x_all"][:, 0],
        results["x_all"][:, 1],
        c="blue",
        label="Converged Trajectory",
    )
    axa.scatter(results["x_all"][:, 0], results["x_all"][:, 1], c="blue")

    axa.set_aspect("equal")
    axa.set_title(r"Position [m]", fontsize=16)
    axa.legend(prop={"size": 15})

    axs[1, 0].plot(
        results["times"], np.linalg.norm(results["x_all"][:, 2:4], axis=1), c="blue"
    )
    axs[1, 0].scatter(
        results["times"], np.linalg.norm(results["x_all"][:, 2:4], axis=1), c="blue"
    )
    axs[1, 0].axhline(y=params["vehicle_v_max"], c="black", linestyle="dashed")
    axs[1, 0].axhline(y=params["vehicle_v_max_evnt"], c="red", linestyle="dashed")
    fsbl_idx = np.argwhere(
        (
            np.linalg.norm(results["x_all"][:, 0:2].T - params["p_w"][0:2, :], axis=0)
            - params["p_w"][2, 0]
        )
        <= 0
    )
    axs[1, 0].axvline(x=results["times"][fsbl_idx[0, 0]], c="red", linestyle="dashed")
    axs[1, 0].set_xlabel(r"time", fontsize=16)
    axs[1, 0].set_ylabel("Speed [m/s]", fontsize=14)

    x_time = np.dstack((results["times"][:-1], results["times"][1:])).reshape(-1)
    axs[1, 1].plot(
        x_time,
        np.linalg.norm(results["u_all"][:, 0:2, :], axis=1).reshape(-1, order="F"),
        c="blue",
    )
    axs[1, 1].scatter(
        x_time,
        np.linalg.norm(results["u_all"][:, 0:2, :], axis=1).reshape(-1, order="F"),
        c="blue",
    )
    axs[1, 1].axhline(y=params["vehicle_a_max"], c="black", linestyle="dashed")
    axs[1, 1].set_xlabel(r"time", fontsize=16)
    axs[1, 1].set_ylabel("Acceleration [m/s$^2$]", fontsize=14)

    if save_fig:
        fig.savefig("figs/Num_sim." + fig_format, bbox_inches="tight", dpi=fig_png_dpi)
