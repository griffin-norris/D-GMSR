import numpy as np
import typing as T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import LineCollection
import matplotlib.patches as patches


def rl_plot(results, params, rl_animations, save_fig, fig_format, fig_png_dpi):

    def rl_oth_fcn(k_oth_sim, fig=None, axs=None):

        if not (rl_animations):
            fig, axs = plt.subplots(
                5, 2, gridspec_kw={"height_ratios": [0.001, 2, 2, 2, 2]}, figsize=(9, 8)
            )

            for ax in axs[0, :]:
                ax.remove()

        k_oth_sim_s = int((k_oth_sim - 1) / (params["t_scp"] / params["dt"]) + 1)

        x_time = np.dstack((results["times"][:-1], results["times"][1:])).reshape(-1)
        x_time_2 = results["times"]

        if rl_animations:
            iii = 0
            jjj = 1
        else:
            iii = 1
            jjj = 0

        ax0 = axs[iii, jjj]
        ax1 = axs[iii, jjj + 1]
        iii += 1
        ax2 = axs[iii, jjj]
        ax3 = axs[iii, jjj + 1]
        iii += 1
        ax4 = axs[iii, jjj]
        ax5 = axs[iii, jjj + 1]
        iii += 1
        ax6 = axs[iii, jjj]
        ax7 = axs[iii, jjj + 1]

        gs_ign = int(params["t_scp"] / params["dt"])
        t_time_gs = results["times"][:]
        t_time_gs_s = x_time_scat[:]

        K0 = results["u_all"][:, 0, :].shape[1]
        K1 = results["x_all"][:, 3:5].shape[0]
        K2 = results["x_nmpc_all"][0, :, 3:5].shape[0]

        temp00 = np.zeros((2, K0))
        temp01 = np.zeros((2, K0))
        temp02 = np.zeros((2, K0))
        temp1 = np.zeros((2, K0))
        temp2 = np.zeros((K1))
        temp2s = np.zeros((K2))
        temp3 = np.zeros((K1))
        temp3s = np.zeros((K2))
        temp4 = np.zeros((K1))
        temp4s = np.zeros((K2))
        temp5 = np.zeros((K1))
        temp5s = np.zeros((K2))
        temp6 = np.zeros((K1))
        temp6s = np.zeros((K2))
        temp7 = np.zeros((K1))
        temp7s = np.zeros((K2))

        temp00[:, :] = None
        temp01[:, :] = None
        temp02[:, :] = None
        temp1[:, :] = None
        temp2[:] = None
        temp2s[:] = None
        temp3[:] = None
        temp3s[:] = None
        temp4[:] = None
        temp4s[:] = None
        temp5[:] = None
        temp5s[:] = None
        temp6[:] = None
        temp6s[:] = None
        temp7[:] = None
        temp7s[:] = None

        temp00[:, :k_oth_sim] = results["u_all"][:, 0, :k_oth_sim]
        temp01[:, :k_oth_sim] = results["u_all"][:, 1, :k_oth_sim]
        temp02[:, :k_oth_sim] = results["u_all"][:, 2, :k_oth_sim]
        temp1[:, :k_oth_sim] = 180 / np.pi * results["u_all"][:, 3, :k_oth_sim]
        temp2[:k_oth_sim] = np.linalg.norm(results["x_all"][:k_oth_sim, 3:5], axis=1)
        temp2s[:k_oth_sim_s] = np.linalg.norm(
            results["x_nmpc_all"][0, :k_oth_sim_s, 3:5], axis=1
        )
        temp3[:k_oth_sim] = 180 / np.pi * results["x_all"][:k_oth_sim, 7]
        temp3s[:k_oth_sim_s] = 180 / np.pi * results["x_nmpc_all"][0, :k_oth_sim_s, 7]
        temp4[:k_oth_sim] = 180 / np.pi * results["x_all"][:k_oth_sim, 5]
        temp4s[:k_oth_sim_s] = 180 / np.pi * results["x_nmpc_all"][0, :k_oth_sim_s, 5]
        temp5[:k_oth_sim] = 180 / np.pi * results["x_all"][:k_oth_sim, 6]
        temp5s[:k_oth_sim_s] = 180 / np.pi * results["x_nmpc_all"][0, :k_oth_sim_s, 6]

        gs_c = (
            180
            / np.pi
            * np.arctan(results["x_all"][:, 2].copy() / results["x_all"][:, 1].copy())
        )

        gs_n = (
            180
            / np.pi
            * np.arctan2(
                results["x_nmpc_all"][0, :k_oth_sim_s, 2].copy(),
                results["x_nmpc_all"][0, :k_oth_sim_s, 1].copy(),
            )
        )

        temp6[: min(K1 - gs_ign, k_oth_sim)] = gs_c[: min(K1 - gs_ign, k_oth_sim)]
        temp6s[: min(K2 - 1, k_oth_sim_s)] = gs_n[: min(K2 - 1, k_oth_sim_s)]

        temp7[:k_oth_sim] = results["x_all"][:k_oth_sim, 0]
        temp7s[:k_oth_sim_s] = results["x_nmpc_all"][0, :k_oth_sim_s, 0]

        temp0_min = results["u_all"][:, 0:3, :].min() * N_2_kN - 100
        temp0_max = results["u_all"][:, 0:3, :].max() * N_2_kN + 100
        temp1_min = 180 / np.pi * results["u_all"][:, 3, :].min() - 2
        temp1_max = 180 / np.pi * results["u_all"][:, 3, :].max() + 2
        temp2_min = np.linalg.norm(results["x_all"][:, 3:5], axis=1).min() - 5
        temp2_max = np.linalg.norm(results["x_all"][:, 3:5], axis=1).max() + 5
        temp3_min = 180 / np.pi * results["x_all"][:, 7].min() - 2
        temp3_max = 180 / np.pi * results["x_all"][:, 7].max() + 2
        temp4_min = 180 / np.pi * results["x_all"][:, 5].min() - 2
        temp4_max = 180 / np.pi * results["x_all"][:, 5].max() + 2
        temp5_min = 180 / np.pi * results["x_all"][:, 6].min() - 2
        temp5_max = 180 / np.pi * results["x_all"][:, 6].max() + 2
        temp6_min = gs_c.min() - 2
        temp6_max = gs_c.max() + 2
        temp7_min = results["x_all"][:, 0].min() - 100
        temp7_max = results["x_all"][:, 0].max() + 100

        ax0.set_ylim(
            min(params["T_min"] * N_2_kN, temp0_min),
            max(params["T_max"] * N_2_kN, temp0_max),
        )
        ax1.set_ylim(
            min(-params["delta_dot_max"], temp1_min),
            max(params["delta_dot_max"], temp1_max),
        )
        ax2.set_ylim(min(0, temp2_min), max(params["spd_trig"], temp2_max))
        ax3.set_ylim(
            min(-180 / np.pi * params["delta_max"], temp3_min),
            max(180 / np.pi * params["delta_max"], temp3_max),
        )
        ax4.set_ylim(
            min(-180 / np.pi * params["theta_trig_aft"], temp4_min),
            max(180 / np.pi * params["theta_trig_aft"], temp4_max),
        )
        ax5.set_ylim(
            min(-180 / np.pi * params["omega_trig_aft"], temp5_min),
            max(180 / np.pi * params["omega_trig_aft"], temp5_max),
        )
        ax6.set_ylim(
            min(-180 / np.pi * params["gs_trig_aft"], temp6_min),
            max(180 / np.pi * params["gs_trig_aft"], temp6_max),
        )
        ax7.set_ylim(min(params["m_f"], temp7_min) / 1e3, max(0, temp7_max) / 1e3)

        # -----------------------------------------------------------------------------------------------

        ax0.plot(
            x_time,
            N_2_kN * temp00.reshape(-1, order="F"),
            c="red",
            lw=3.0,
            label="Engine - 1",
        )
        ax0.plot(
            x_time,
            N_2_kN * temp01.reshape(-1, order="F"),
            c="orange",
            lw=2.5,
            linestyle="dashed",
            label="Engine - 2",
        )
        ax0.plot(
            x_time,
            N_2_kN * temp02.reshape(-1, order="F"),
            c="blue",
            lw=1.0,
            linestyle="dotted",
            label="Engine - 3",
        )

        if x_time_2[k_oth_sim - 1] >= t_v_trig:
            ax0.axvline(
                x=t_v_trig,
                c=c_t_v_trig,
                linestyle="dashed",
                label="$t_{{v}_{\mathrm{trigger}}}$",
            )

        ax0.axhline(y=N_2_kN * params["T_max"], c=c_up_lim, linestyle="dashed")
        ax0.axhline(y=N_2_kN * params["T_min"], c=c_low_lim, linestyle="dashed")

        ax0.set_xlim(0, x_time_2[-1])
        ax0.set_ylabel("Thrust, $T$ [kN]", fontsize=y_fs)

        # -----------------------------------------------------------------------------------------------

        ax1.plot(x_time, temp1.reshape(-1, order="F"), c=c_plt)
        ax1.axhline(
            y=180 / np.pi * params["delta_dot_max"],
            c=c_up_lim,
            linestyle="dashed",
            label="Upper bounds",
        )
        ax1.axhline(
            y=-180 / np.pi * params["delta_dot_max"],
            c=c_low_lim,
            linestyle="dashed",
            label="Lower bounds",
        )

        ax1.set_xlim(0, x_time_2[-1])
        ax1.set_ylabel("Gimbal rate, $\dot{\delta}$ [deg s$^{-1}$]", fontsize=y_fs)

        # -----------------------------------------------------------------------------------------------

        ax2.plot(x_time_2, temp2, c=c_plt)
        ax2.scatter(x_time_scat, temp2s, s=scatter_sc, c=c_node)
        if x_time_2[k_oth_sim - 1] >= t_h_trig:
            ax2.axvline(
                x=t_h_trig,
                c=c_trig,
                linestyle="dashed",
                label="$t_{r_{x_{\mathrm{trigger}}}}$",
            )
            ax2.plot(
                [t_h_trig, x_time_2[k_oth_sim - 1]],
                [params["spd_trig_aft"], params["spd_trig_aft"]],
                c=c_up_lim,
                linestyle="dashed",
            )

        if x_time_2[k_oth_sim - 1] >= t_v_trig:
            ax2.axvline(x=t_v_trig, c=c_t_v_trig, linestyle="dashed")
            ax2.plot(
                [t_v_trig, min(x_time_2[k_oth_sim - 1], t_h_trig)],
                [params["spd_trig"], params["spd_trig"]],
                c=c_up_lim,
                linestyle="dashed",
            )

        ax2.set_xlim(0, x_time_2[-1])
        ax2.set_ylabel("Speed, $\| v \|_2$ [m s$^{-1}$]", fontsize=y_fs)

        # -----------------------------------------------------------------------------------------------

        ax3.plot(x_time_2, temp3, c=c_plt)
        ax3.scatter(x_time_scat, temp3s, s=scatter_sc, c=c_node)

        ax3.plot(
            [0, min(x_time_2[k_oth_sim - 1], t_h_trig)],
            [180 / np.pi * params["delta_max"], 180 / np.pi * params["delta_max"]],
            c=c_up_lim,
            linestyle="dashed",
        )

        ax3.plot(
            [0, min(x_time_2[k_oth_sim - 1], t_h_trig)],
            [-180 / np.pi * params["delta_max"], -180 / np.pi * params["delta_max"]],
            c=c_low_lim,
            linestyle="dashed",
        )

        if x_time_2[k_oth_sim - 1] >= t_h_trig:
            ax3.plot(
                [t_h_trig, x_time_2[k_oth_sim - 1]],
                [
                    180 / np.pi * params["delta_trig_aft"],
                    180 / np.pi * params["delta_trig_aft"],
                ],
                c=c_up_lim,
                linestyle="dashed",
            )
            ax3.plot(
                [t_h_trig, x_time_2[k_oth_sim - 1]],
                [
                    -180 / np.pi * params["delta_trig_aft"],
                    -180 / np.pi * params["delta_trig_aft"],
                ],
                c=c_low_lim,
                linestyle="dashed",
            )
            ax3.axvline(x=t_h_trig, c=c_trig, linestyle="dashed")

        ax3.set_xlim(0, x_time_2[-1])
        ax3.set_ylabel("Gimbal angle, $\delta$ [deg]", fontsize=y_fs)

        # -----------------------------------------------------------------------------------------------

        ax4.plot(x_time_2, temp4, c=c_plt)
        ax4.scatter(x_time_scat, temp4s, s=scatter_sc, c=c_node)

        if x_time_2[k_oth_sim - 1] >= t_h_trig:
            ax4.plot(
                [t_h_trig, x_time_2[k_oth_sim - 1]],
                [
                    180 / np.pi * params["theta_trig_aft"],
                    180 / np.pi * params["theta_trig_aft"],
                ],
                c=c_up_lim,
                linestyle="dashed",
            )
            ax4.plot(
                [t_h_trig, x_time_2[k_oth_sim - 1]],
                [
                    -180 / np.pi * params["theta_trig_aft"],
                    -180 / np.pi * params["theta_trig_aft"],
                ],
                c=c_low_lim,
                linestyle="dashed",
            )

            ax4.axvline(x=t_h_trig, c=c_trig, linestyle="dashed")

        ax4.set_xlim(0, x_time_2[-1])
        ax4.set_ylabel("Tilt angle, $\\theta$ [deg]", fontsize=y_fs)

        # -----------------------------------------------------------------------------------------------

        ax5.plot(x_time_2, temp5, c=c_plt)
        ax5.scatter(x_time_scat, temp5s, s=scatter_sc, c=c_node)

        if x_time_2[k_oth_sim - 1] >= t_h_trig:
            ax5.plot(
                [t_h_trig, x_time_2[k_oth_sim - 1]],
                [
                    180 / np.pi * params["omega_trig_aft"],
                    180 / np.pi * params["omega_trig_aft"],
                ],
                c=c_up_lim,
                linestyle="dashed",
            )
            ax5.plot(
                [t_h_trig, x_time_2[k_oth_sim - 1]],
                [
                    -180 / np.pi * params["omega_trig_aft"],
                    -180 / np.pi * params["omega_trig_aft"],
                ],
                c=c_low_lim,
                linestyle="dashed",
            )
            ax5.axvline(x=t_h_trig, c=c_trig, linestyle="dashed")

        ax5.set_xlim(0, x_time_2[-1])
        ax5.set_ylabel("Angular velocity, $\omega$ [deg s$^{-1}$]", fontsize=y_fs)

        # -----------------------------------------------------------------------------------------------

        ax6.plot(t_time_gs, temp6, c=c_plt)
        ax6.scatter(t_time_gs_s, temp6s, s=scatter_sc, c=c_node)

        if x_time_2[k_oth_sim - 1] >= t_h_trig:
            ax6.plot(
                [t_h_trig, x_time_2[k_oth_sim - 1]],
                [
                    180 / np.pi * params["gs_trig_aft"],
                    180 / np.pi * params["gs_trig_aft"],
                ],
                c=c_up_lim,
                linestyle="dashed",
            )

            ax6.plot(
                [t_h_trig, x_time_2[k_oth_sim - 1]],
                [
                    -180 / np.pi * params["gs_trig_aft"],
                    -180 / np.pi * params["gs_trig_aft"],
                ],
                c=c_low_lim,
                linestyle="dashed",
            )

            ax6.axvline(x=t_h_trig, c=c_trig, linestyle="dashed")

        ax6.set_xlim(0, x_time_2[-1] - 1)
        ax6.set_xlabel("Time [s]", fontsize=x_fs)
        ax6.set_ylabel("Glideslope angle, $\gamma$ [deg]", fontsize=y_fs)

        # -----------------------------------------------------------------------------------------------

        ax7.plot(x_time_2, temp7 / 1e3, c=c_plt)
        ax7.scatter(x_time_scat, temp7s / 1e3, s=scatter_sc, c=c_node)
        ax7.axhline(y=params["m_f"] / 1e3, c=c_low_lim, linestyle="dashed")
        ax7.set_xlim(0, x_time_2[-1])
        ax7.set_xlabel("Time [s]", fontsize=x_fs)
        ax7.set_ylabel("Mass, $m$ [10$^{3}$ kg]", fontsize=y_fs)

        # -----------------------------------------------------------------------------------------------

        lines_labels = [axs.get_legend_handles_labels() for axs in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        if rl_animations:
            fig.legend(
                lines, labels, loc="upper left", ncol=9, mode="expand", fontsize=leg_fc
            )
        else:
            fig.legend(
                lines,
                labels,
                loc="upper left",
                ncol=9,
                mode="expand",
                fontsize=leg_fc - 2,
            )

        if not (rl_animations):
            if save_fig:
                fig.savefig(
                    "figs/rl_oth." + fig_format, bbox_inches="tight", dpi=fig_png_dpi
                )

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    x_fs = 14
    y_fs = 11
    leg_fc = 13

    xy_fs = 14

    N_2_kN = 1e-3
    scatter_sc = 15.0

    c_up_lim = "green"
    c_low_lim = "purple"
    c_trig = "red"
    c_trig_alt = "orange"
    c_node = "black"
    c_t_v_trig = "black"
    c_gs = "green"
    c_plt = "blue"

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    x_time_scat = results["times"][:: int(params["t_scp"] / params["dt"])]

    idx_h_trig_0 = np.argmax(
        results["x_nmpc_all"][0, :, 1] - params["alt_trig"] < -1e-4
    )
    t_h_trig = x_time_scat[idx_h_trig_0]

    idx_v_trig_0 = np.argmax(
        np.linalg.norm(results["x_nmpc_all"][0, :, 3:5], axis=1) - params["spd_trig"]
        < -1e-4
    )
    t_v_trig = x_time_scat[idx_v_trig_0]

    # -----------------------------------------------------------------------------------------------

    spd_norm = np.linalg.norm(results["x_all"][:, 3:5], axis=1)
    points = np.array([results["x_all"][:, 2], results["x_all"][:, 1]]).T.reshape(
        -1, 1, 2
    )
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # -----------------------------------------------------------------------------------------------

    if rl_animations:
        rl_rocket_list = [0]
        rl_anim_list = range(1, results["x_all"].shape[0])
    else:
        # 8-8-8-16-24-24-24-28-28-32
        rl_rocket_list = [0, 8, 16, 24, 40, 64, 88, 112, 140, 168, 199]
        rl_anim_list = [results["x_all"].shape[0]]

    # -----------------------------------------------------------------------------------------------

    for rl_anim_k in rl_anim_list:

        if rl_animations:
            print("Figure: {}/{}".format(rl_anim_k, results["x_all"].shape[0] - 1))
            fig, axs = plt.subplots(
                5,
                3,
                gridspec_kw={
                    "height_ratios": [0.001, 2, 2, 2, 2],
                    "width_ratios": [2, 2, 2],
                },
                figsize=(13, 8),
            )

            gs = axs[1, 0].get_gridspec()
            for ax in axs[1:, 0]:
                ax.remove()
            for ax in axs[0, :]:
                ax.remove()
            axa = fig.add_subplot(gs[1:, 0])

        else:
            fig, axa = plt.subplots(1, 1, figsize=(9, 8))

        if rl_anim_k > 0:
            lc = LineCollection(
                segments[:rl_anim_k, :],
                cmap=plt.cm.rainbow,
                norm=plt.Normalize(0.0, spd_norm.max()),
                array=spd_norm[: rl_anim_k + 1],
                lw=3,
            )
            axa.add_collection(lc)
            cbar = fig.colorbar(lc, aspect=50, pad=0.01)
            cbar.set_label(
                "Speed, $\| v \|_2$ [m s$^{-1}$]", fontsize=xy_fs, labelpad=5
            )

        # Trust
        rt0 = np.array([[-params["l_r"] / 2], [-params["l_cm"]]])

        for k in rl_rocket_list:

            if rl_animations:
                k = rl_anim_k

            ang = results["x_all"][k, 5]
            rt_mt = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

            rt = rt_mt @ rt0
            rect = patches.Rectangle(
                (results["x_all"][k, 2] + rt[0, 0], results["x_all"][k, 1] + rt[1, 0]),
                2 * params["l_r"] / 2,
                params["l_h"],
                angle=ang * 180 / np.pi,
                linewidth=1,
                edgecolor="black",
                facecolor="gray",
                zorder=10,
            )
            axa.add_patch(rect)

            t_mag = np.zeros(results["u_all"][:, 0:3, :].shape[-1] + 1)
            t_mag[:-1] = ((results["u_all"][:, 0:3, :].sum(axis=1)))[0, :] / 2e5 * 3
            t_mag[-1] = ((results["u_all"][:, 0:3, :].sum(axis=1)))[1, -1] / 2e5 * 3
            t_mag[: int(t_v_trig)] /= 3

            ang_2 = results["x_all"][k, 7]
            R_IB = np.array(
                [[np.cos(ang_2), np.sin(ang_2)], [-np.sin(ang_2), np.cos(ang_2)]]
            )

            rts = rt_mt @ np.array([[0.0], [-params["l_cm"]]])
            rte = rts + R_IB.T @ rt_mt @ np.array([[0.0], [-t_mag[k]]])

            if k == results["x_all"].shape[0] - 1:
                ada = np.linalg.norm(results["u_all"][1, 1:3, -1])
            else:
                ada = np.linalg.norm(results["u_all"][0, 1:3, k])

            if ada >= 1e-3:

                if (k == 0) and not (rl_animations):
                    axa.plot(
                        results["x_all"][k, 2] + np.array([rts[0, 0], rte[0, 0]]),
                        results["x_all"][k, 1] + np.array([rts[1, 0], rte[1, 0]]),
                        lw=3.0,
                        c="red",
                        zorder=11,
                        label="Engine - 1",
                    )

                    axa.plot(
                        results["x_all"][k, 2] + np.array([rts[0, 0], rte[0, 0]]),
                        results["x_all"][k, 1] + np.array([rts[1, 0], rte[1, 0]]),
                        lw=1.5,
                        c="orange",
                        zorder=11,
                        label="Engine - 2",
                    )

                    axa.plot(
                        results["x_all"][k, 2] + np.array([rts[0, 0], rte[0, 0]]),
                        results["x_all"][k, 1] + np.array([rts[1, 0], rte[1, 0]]),
                        lw=0.5,
                        c="blue",
                        zorder=11,
                        label="Engine - 3",
                    )

                else:
                    axa.plot(
                        results["x_all"][k, 2] + np.array([rts[0, 0], rte[0, 0]]),
                        results["x_all"][k, 1] + np.array([rts[1, 0], rte[1, 0]]),
                        lw=3.0,
                        c="red",
                        zorder=11,
                    )

                    axa.plot(
                        results["x_all"][k, 2] + np.array([rts[0, 0], rte[0, 0]]),
                        results["x_all"][k, 1] + np.array([rts[1, 0], rte[1, 0]]),
                        lw=2.0,
                        c="orange",
                        zorder=11,
                    )

                    axa.plot(
                        results["x_all"][k, 2] + np.array([rts[0, 0], rte[0, 0]]),
                        results["x_all"][k, 1] + np.array([rts[1, 0], rte[1, 0]]),
                        lw=0.75,
                        c="blue",
                        zorder=11,
                    )

            else:
                axa.plot(
                    results["x_all"][k, 2] + np.array([rts[0, 0], rte[0, 0]]),
                    results["x_all"][k, 1] + np.array([rts[1, 0], rte[1, 0]]),
                    lw=1.0,
                    c="red",
                    zorder=0,
                )

        axa.axhline(
            y=params["alt_trig"],
            c=c_trig_alt,
            linestyle="dashed",
            label="$r_{x_{\mathrm{trigger}}}$",
        )

        # Glideslope
        triangle1 = patches.Polygon(
            (
                (0.0, 0.0),
                (
                    params["alt_trig"] * np.tan(params["gs_trig_aft"]),
                    params["alt_trig"],
                ),
                (
                    -params["alt_trig"] * np.tan(params["gs_trig_aft"]),
                    params["alt_trig"],
                ),
            ),
            fc=(0, 1, 0, 0.5),
            ec=(0, 1, 0, 1),
            lw=1,
            label="Glideslope",
        )
        axa.add_artist(triangle1)

        axa.set_aspect("equal")
        axa.set_xlim(-75, 160)
        axa.set_ylim(-75, 525)
        axa.set_xlabel("Downrange, $r^z$ [m]", fontsize=xy_fs)
        axa.set_ylabel("Altitude, $r^x$ [m]", fontsize=xy_fs)
        if not (rl_animations):
            axa.legend(prop={"size": leg_fc}, loc="upper left")

        if not (rl_animations):
            if save_fig:
                fig.savefig(
                    "figs/rl_pos." + fig_format, bbox_inches="tight", dpi=fig_png_dpi
                )

        if rl_animations:
            rl_oth_fcn(k_oth_sim=rl_anim_k, fig=fig, axs=axs[1:, :])

            newpath = r"sim"
            if not os.path.exists(newpath):
                os.makedirs(newpath)

            fig.savefig(
                "sim/rl_pos_" + str(rl_anim_k).zfill(3) + "." + "png",
                bbox_inches="tight",
                dpi=int(fig_png_dpi / 2),
            )
            plt.close(fig)
        else:
            rl_oth_fcn(k_oth_sim=rl_anim_k)

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
