import numpy as np
import typing as T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from itertools import count, groupby


def plt_drone_fcn(ax, center, z_dir, length_drone, head_angle):

    def cyl(ax, p0, p1, rad_drone, clr=None, clr2=None):

        # Vector in direction of axis
        v = p1 - p0

        # Find magnitude of vector
        mag = np.linalg.norm(v + 1e-6)

        # Unit vector in direction of axis
        v = v / mag

        # Make some vector not in the same direction as v
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])

        # Make vector perpendicular to v
        n1 = np.cross(v, not_v)
        # normalize n1
        n1 /= np.linalg.norm(n1 + 1e-6)

        # Make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)

        # Surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0, mag, 2)
        theta = np.linspace(0, 2 * np.pi, 100)
        rsample = np.linspace(0, rad_drone, 2)

        # Use meshgrid to make 2d arrays
        t, theta2 = np.meshgrid(t, theta)

        rsample, theta = np.meshgrid(rsample, theta)

        # Generate coordinates for surface
        # "Tube"
        X, Y, Z = [
            p0[i]
            + v[i] * t
            + rad_drone * np.sin(theta2) * n1[i]
            + rad_drone * np.cos(theta2) * n2[i]
            for i in [0, 1, 2]
        ]
        # "Bottom"
        X2, Y2, Z2 = [
            p0[i]
            + rsample[i] * np.sin(theta) * n1[i]
            + rsample[i] * np.cos(theta) * n2[i]
            for i in [0, 1, 2]
        ]
        # "Top"
        X3, Y3, Z3 = [
            p0[i]
            + v[i] * mag
            + rsample[i] * np.sin(theta) * n1[i]
            + rsample[i] * np.cos(theta) * n2[i]
            for i in [0, 1, 2]
        ]

        ax.plot_surface(X, Y, Z, color=clr, zorder=9)
        ax.plot_surface(X2, Y2, Z2, color=clr, zorder=9)
        ax.plot_surface(X3, Y3, Z3, color=clr, zorder=9)

        if clr2:
            phi = np.linspace(0, 2 * np.pi, 50)
            theta = np.linspace(0, np.pi, 25)

            dx = 3 * rad_drone * np.outer(np.cos(phi), np.sin(theta))
            dy = 3 * rad_drone * np.outer(np.sin(phi), np.sin(theta))
            dz = 3 * rad_drone * np.outer(np.ones(np.size(phi)), np.cos(theta))

            ax.plot_surface(
                p1[0] + dx,
                p1[1] + dy,
                p1[2] + dz,
                cstride=1,
                rstride=1,
                color=clr2,
                zorder=10,
            )
            ax.plot_surface(
                p0[0] + dx,
                p0[1] + dy,
                p0[2] + dz,
                cstride=1,
                rstride=1,
                color=clr2,
                zorder=10,
            )

    # Rodrigues' rotation formula
    def rotate_vec(v, d, alpha):
        ada = (
            v * np.cos(alpha)
            + np.cross(d, v) * np.sin(alpha)
            + d * np.dot(d, v) * (1 - np.cos(alpha))
        )
        return ada

    z_dir = z_dir / np.linalg.norm(z_dir + 1e-6)
    rad_drone = length_drone * 0.02

    l1_axis = rotate_vec(head_angle, z_dir, np.pi / 4)
    p0 = center - l1_axis * length_drone / 2
    p1 = center + l1_axis * length_drone / 2

    l2_axis = rotate_vec(l1_axis, z_dir, np.pi / 2)
    p2 = center - l2_axis * length_drone / 2
    p3 = center + l2_axis * length_drone / 2

    # Body
    cyl(ax, p0, p1, rad_drone, clr="black", clr2="yellow")
    cyl(ax, p2, p3, rad_drone, clr="black", clr2="yellow")

    # Head
    p6 = center
    p7 = center + head_angle * length_drone / 4
    cyl(ax, p6, p7, rad_drone / 1.5, clr="gray")

    p8 = center
    p9 = center + z_dir * length_drone / 2

    # Trust
    cyl(ax, p8, p9, rad_drone * 0.8, clr="red")


def qf_plot(results, params, save_fig, fig_format, fig_png_dpi):

    xy_fs = 15
    scatter_sc = 15
    c_up_lim = "green"
    c_low_lim = "purple"
    c_trig = "red"
    c_plt = "blue"

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection="3d", computed_zorder=False)

    spd_norm = np.linalg.norm(results["x_all"][:, 3:6], axis=1)

    points = np.array(
        [results["x_all"][:, 0], results["x_all"][:, 1], results["x_all"][:, 2]]
    ).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(
        segments,
        cmap=plt.cm.rainbow,
        norm=plt.Normalize(0.0, spd_norm.max()),
        array=spd_norm,
        lw=3,
        zorder=-2,
    )

    ax.add_collection(lc)
    cbar = fig.colorbar(lc, aspect=50, pad=-0.27, shrink=0.75, orientation="horizontal")
    cbar.set_label("Speed [m s$^{-1}$]", fontsize=xy_fs, labelpad=5)

    # Plot the charging area
    U3d, s3d, rotation = np.linalg.svd(np.eye(3))
    radii = params["p_w"][3] / s3d

    u = np.linspace(0.0, 2.0 * np.pi, 15)
    v = np.linspace(0.0, 1.0 * np.pi, 15)

    x_sp = radii[0] * np.outer(np.cos(u), np.sin(v))
    y_sp = radii[1] * np.outer(np.sin(u), np.sin(v))
    z_sp = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x_sp)):
        for j in range(len(x_sp)):
            [x_sp[i, j], y_sp[i, j], z_sp[i, j]] = (
                np.dot([x_sp[i, j], y_sp[i, j], z_sp[i, j]], rotation)
                + params["p_w"][0:3, 0]
            )

    ax.plot_wireframe(
        x_sp,
        y_sp,
        z_sp,
        rstride=2,
        cstride=2,
        edgecolor="black",
        linewidth=0.1,
        zorder=11,
    )

    # Plot the quadrotor
    for k in np.linspace(0, results["x_all"].shape[0] - 1, 12, dtype=int):
        if k == results["x_all"].shape[0] - 1:
            c_inp = results["u_all"][1, :, -1]
        else:
            c_inp = results["u_all"][0, :, k]

        plt_drone_fcn(
            ax=ax,
            center=results["x_all"][k, 0:3],
            z_dir=c_inp,
            length_drone=0.4,
            head_angle=np.array([1.0, 0.0, 0.0]),
        )

    # Plot
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")

    xy_fs = 15
    ax.set_xlabel("$r^x$ [m]", size=xy_fs)
    ax.set_ylabel("$r^y$ [m]", size=xy_fs)
    ax.set_zlabel("$r^z$ [m]", size=xy_fs)

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 11
    ax.zaxis.labelpad = 0.5

    xy_fs = 14
    ax.xaxis.set_tick_params(labelsize=xy_fs)
    ax.yaxis.set_tick_params(labelsize=xy_fs)
    ax.zaxis.set_tick_params(labelsize=xy_fs)
    ax.set_zticks(np.arange(0, 2.5, 0.5))

    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(-0.2, 2.2)
    ax.set_zlim(0.0, 2.2)

    ax.view_init(10, 220)
    ax.set_aspect("equal")
    if save_fig:
        fig.savefig("figs/qf_pos0." + fig_format, bbox_inches="tight", dpi=fig_png_dpi)

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    fig, axs = plt.subplots(
        3, 2, gridspec_kw={"height_ratios": [3, 3, 0.001]}, figsize=(9, 5)
    )

    for ax in axs[-1, :]:
        ax.remove()

    spd_norm = np.linalg.norm(results["x_all"][:, 3:6], axis=1)

    axa = axs[0, 0]
    x_time = np.dstack((results["times"][:-1], results["times"][1:])).reshape(-1)
    axa.plot(
        x_time,
        np.linalg.norm(results["u_all"][:, 0:3, :], axis=1).reshape(-1, order="F"),
        c="blue",
    )
    axa.axhline(y=params["vehicle_T_max"], c=c_up_lim, linestyle="dashed")
    axa.set_ylabel("Thrust, $T$ [N]", fontsize=xy_fs)

    axa = axs[0, 1]
    tilt_angle = np.zeros((1 + results["u_all"][0, 0, :].shape[0]))
    tilt_angle[:-1] = np.arccos(
        results["u_all"][0, 2, :] / np.linalg.norm(results["u_all"][0, 0:3, :], axis=0)
    )
    tilt_angle[-1] = np.arccos(
        results["u_all"][1, 2, -1] / np.linalg.norm(results["u_all"][1, 0:3, -1])
    )
    axa.plot(results["times"][:], 180 / np.pi * tilt_angle, c=c_plt)
    axa.axhline(
        y=180 / np.pi * params["vehicle_theta_max"], c=c_up_lim, linestyle="dashed"
    )
    axa.set_ylabel("Tilt angle, $\\theta$ [deg]", fontsize=xy_fs)

    fsbl_idx = np.argwhere(
        (
            np.linalg.norm(
                results["x_nmpc_all"][0, :, 0:3].T - params["p_w"][0:3, :], axis=0
            )
            - params["p_w"][3, 0]
        )
        <= 0
    )
    c = count()
    groups = groupby(fsbl_idx, lambda x: x - next(c))
    all_conseq = (list(g) for _, g in groups)
    fsbl_idx = np.vstack(max(all_conseq, key=len))
    mf0 = results["times"][:: int(params["t_scp"] / params["dt"])][fsbl_idx[0, 0]]
    mf1 = results["times"][:: int(params["t_scp"] / params["dt"])][fsbl_idx[-1, 0]]

    axa = axs[1, 0]
    axa.plot(results["times"], spd_norm, c=c_plt)
    spd_node = np.linalg.norm(results["x_nmpc_all"][0, :, 3:6], axis=1)
    axa.scatter(
        results["times"][:: int(params["t_scp"] / params["dt"])],
        spd_node,
        s=scatter_sc,
        c="black",
    )
    axa.axvline(x=mf0, c="red", linestyle="dashed", label="t$_{\mathrm{start}}$")
    axa.axvline(x=mf1, c="orange", linestyle="dashed", label="t$_{\mathrm{end}}$")

    axa.plot(
        [0, mf1],
        [params["vehicle_v_max_evnt"], params["vehicle_v_max_evnt"]],
        c=c_up_lim,
        linestyle="dashed",
        label="Upper bounds",
    )

    axa.plot(
        [mf1, params["t_f"]],
        [params["vehicle_v_max"], params["vehicle_v_max"]],
        c=c_up_lim,
        linestyle="dashed",
    )

    axa.set_xlabel(r"time", fontsize=xy_fs)
    axa.set_ylabel("Speed, $v$ [m/s]", fontsize=xy_fs)

    axa = axs[1, 1]
    dist_to_stat = np.linalg.norm(
        (results["x_all"][:, 0:3] - params["p_w"][0:3, 0]), axis=1
    )
    axa.plot(results["times"], dist_to_stat, c=c_plt)

    dist_to_stat = np.linalg.norm(
        (results["x_nmpc_all"][0, :, 0:3] - params["p_w"][0:3, 0]), axis=1
    )
    axa.scatter(
        results["times"][:: int(params["t_scp"] / params["dt"])],
        dist_to_stat,
        c="black",
        s=scatter_sc,
    )

    axa.axvline(x=mf0, c="red", linestyle="dashed")
    axa.axvline(x=mf1, c="orange", linestyle="dashed")
    axa.axhline(y=params["p_w"][3], c=c_up_lim, linestyle="dashed")
    axa.set_xlabel(r"time", fontsize=xy_fs)
    axa.set_ylabel(
        "Distance to the station's \n center, $\| r - r_w \|_2$ [m]", fontsize=xy_fs
    )

    lines_labels = [axs.get_legend_handles_labels() for axs in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines, labels, loc=(0.262, 0.006), ncol=6, fontsize=15
    )  # , mode = "expand")

    if save_fig:
        fig.savefig("figs/qf_oth." + fig_format, bbox_inches="tight", dpi=fig_png_dpi)
