def lm_params_fcn():
    params = dict()

    # Problem
    params["K"] = 9
    params["wp"] = np.array([[1.25], [-3], [0.2]])

    # STL
    params["stl_eps"] = 1e-8
    params["stl_p"] = 1
    params["stl_w"] = np.ones(params["K"])
    params["f_func_ssr"], params["A_func_ssr"] = ssr_sympy(k=25, K=params["K"])

    # Prox-linear
    params["N_ite"] = 10
    params["w_tr0"] = 0.8
    params["r0"] = 0.01
    params["r1"] = 0.1
    params["r2"] = 0.9
    params["adaptive_step"] = True

    return params


def RUN_LM(params):

    def f_wp_evnt(xk, uk, dx, du, params, npy=True):

        dist_n = -(
            -params["wp"][2, 0] + np.linalg.norm(xk - params["wp"][:2, :], axis=0)
        )
        if params["stl_type"] == "D-SSR":
            f0, g1 = (
                params["f_func_ssr"](dist_n)[1:, :],
                params["A_func_ssr"](dist_n)[1, :],
            )
        elif params["stl_type"] == "D-GMSR":
            f0, g1 = gmsr_or(
                params["stl_eps"], params["stl_p"], params["stl_w"], dist_n
            )

        direction = -(xk - params["wp"][:2, :]) / dist_n
        g2 = -(np.diag(g1) @ direction.T).T
        if not (npy):
            lin = f0 + cp.sum(cp.multiply(g2, dx))
        else:
            lin = f0 + (g2 * dx).sum()

        return -f0, -lin

    def f_ptr(dx, du, npy=True):
        if npy:
            return np.sum((np.linalg.norm(dx, axis=0)) ** 2) + np.sum(
                (np.linalg.norm(du, axis=0)) ** 2
            )
        else:
            return cp.sum((cp.norm(dx, axis=0)) ** 2) + cp.sum(
                (cp.norm(du, axis=0)) ** 2
            )

    def solve_cvx(X, U, w_tr, params):

        dX = cp.Variable((2, params["K"]))
        dU = cp.Variable((2, params["K"] - 1))

        _, lin = f_wp_evnt(X, U, dX, dU, params, npy=False)

        mdl_obj = cp.Minimize(lin)
        ptr_obj = cp.Minimize(w_tr * f_ptr(dX, dU, npy=False))

        obj = mdl_obj + ptr_obj
        cons = [
            (X + dX)[:, 1:] == (X + dX)[:, 0:-1] + (U + dU),
            (X + dX)[:, 0] == np.array([0.0, 0.0]),
            (X + dX)[:, -1] == np.array([8.0, 0.0]),
            cp.norm(U + dU, axis=0) <= 1.5,
        ]

        problem = cp.Problem(obj, cons)

        try:
            problem.solve(solver="ECOS")
        except:
            problem.solve(solver="MOSEK")

        return dX.value, dU.value, ptr_obj.value

    def prox_linear(params):

        w_tr = params["w_tr0"]

        Xk = np.zeros((2, params["K"]))
        Uk = np.zeros((2, params["K"] - 1))

        Xk[0, :] = np.arange(0, params["K"], 1)
        Uk[0, :] = np.ones(params["K"] - 1)

        nl0, _ = f_wp_evnt(Xk, Uk, Xk - Xk, Uk - Uk, params, npy=True)
        f_list = [nl0]
        x_list = [Xk]

        for i in range(params["N_ite"]):
            while True:

                dX, dU, ptr_val = solve_cvx(Xk, Uk, w_tr, params)

                _, lf = f_wp_evnt(Xk, Uk, dX, dU, params, npy=True)
                nl0, _ = f_wp_evnt(Xk, Uk, Xk - Xk, Uk - Uk, params, npy=True)
                nl1, _ = f_wp_evnt(Xk + dX, Uk + dU, Xk - Xk, Uk - Uk, params, npy=True)

                if params["adaptive_step"]:

                    act_chg = nl0 - nl1
                    prd_chg = nl0 - lf

                    rho = act_chg / (prd_chg - ptr_val)

                    if rho < params["r0"]:
                        w_tr *= 2.0
                    else:

                        Xk = Xk + dX
                        Uk = Uk + dU

                        f_list.append(nl1)
                        x_list.append(Xk)

                        if params["r2"] <= rho:
                            w_tr /= 2.0
                            w_tr = np.maximum(w_tr, 0.1)

                        break
                else:
                    Xk = Xk + dX
                    Uk = Uk + dU

                    f_list.append(nl1)
                    x_list.append(Xk)
                    break

        f_list = -np.vstack(f_list)
        x_list = (np.vstack(x_list)).reshape(-1, 2, params["K"])

        return f_list, x_list

    def plot(params, results):
        fig, axs = plt.subplots(
            nrows=3, ncols=2, gridspec_kw={"height_ratios": [5, 3, 2]}, figsize=(8, 8)
        )

        gs = axs[0, 0].get_gridspec()
        for ax in axs[0, :]:
            ax.remove()
        axa = fig.add_subplot(gs[0, :])

        circle = plt.Circle(
            (params["wp"][0, 0], params["wp"][1, 0]),
            params["wp"][2, 0],
            color="blue",
            clip_on=False,
            fill=False,
        )
        axa.add_patch(circle)

        cc = plt.cm.rainbow(np.linspace(0, 1, params["K"]))

        axa.plot(
            results["x_dgmsr"][0, 0, :],
            results["x_dgmsr"][0, 1, :],
            c="black",
            label="Initial Trajectory",
            linestyle="dotted",
        )
        axa.scatter(
            results["x_dgmsr"][0, 0, :],
            results["x_dgmsr"][0, 1, :],
            c=cc,
            alpha=1,
            s=50,
            zorder=10,
        )

        axa.plot(
            results["x_ssr"][-1, 0, :],
            results["x_ssr"][-1, 1, :],
            c="red",
            label="D-SSR",
            linestyle="dotted",
        )
        axa.scatter(
            results["x_ssr"][-1, 0, :],
            results["x_ssr"][-1, 1, :],
            c=cc,
            alpha=1,
            s=50,
            zorder=10,
        )

        axa.plot(
            results["x_dgmsr"][-1, 0, :],
            results["x_dgmsr"][-1, 1, :],
            c="blue",
            label="D-GMSR",
            linestyle="dotted",
        )
        axa.scatter(
            results["x_dgmsr"][-1, 0, :],
            results["x_dgmsr"][-1, 1, :],
            c=cc,
            alpha=1,
            s=50,
            zorder=10,
        )

        axa.set_aspect("equal")
        axa.set_title("Position [m]", fontsize=16)
        axa.legend(prop={"size": 14})

        color_1 = iter(cm.rainbow(np.linspace(0, 1, params["K"])))
        plt.figure()
        for i in range(params["K"]):
            c = next(color_1)
            all_dist = -(
                -params["wp"][2, 0]
                + np.linalg.norm(
                    results["x_ssr"][:, :, i].T - params["wp"][:2, :], axis=0
                )
            )
            axs[1, 0].plot(all_dist, c=c)
        axs[1, 0].axhline(0, c="black", linestyle="dashed")
        axs[1, 0].set_xlabel(r"iterations", fontsize=16)
        axs[1, 0].set_ylabel("$f(x_k)$", fontsize=16)
        axs[1, 0].set_title("D-SSR", fontsize=16)

        axs[2, 0].plot(results["f_ssr"], c="blue")
        axs[2, 0].axhline(0, c="black", linestyle="dashed")
        axs[2, 0].set_xlabel(r"iterations", fontsize=16)
        axs[2, 0].set_ylabel(
            "$$\\tilde{\\rho}^{ \\bm{F}_{[1:K]} \\varphi} (x,0)$$", fontsize=16
        )

        color_1 = iter(cm.rainbow(np.linspace(0, 1, params["K"])))
        for i in range(params["K"]):
            c = next(color_1)
            all_dist = -(
                -params["wp"][2, 0]
                + np.linalg.norm(
                    results["x_dgmsr"][:, :, i].T - params["wp"][:2, :], axis=0
                )
            )
            axs[1, 1].plot(all_dist, c=c)
        axs[1, 1].axhline(0, c="black", linestyle="dashed")
        axs[1, 1].set_xlabel(r"iterations", fontsize=16)
        axs[1, 1].set_ylabel("$f(x_k)$", fontsize=16)
        axs[1, 1].set_title("D-GMSR", fontsize=16)

        axs[2, 1].plot(results["f_dgmsr"], c="blue")
        axs[2, 1].axhline(0, c="black", linestyle="dashed")
        axs[2, 1].set_xlabel(r"iterations", fontsize=16)
        axs[2, 1].set_ylabel(
            "$\\Gamma^{ \\bm{F}_{[1:K]} \\varphi}_{\\bm{\\epsilon}, \\bm{p}, \\bm{w}} (x,0)$",
            fontsize=16,
        )

        if save_fig:
            fig.savefig(
                "figs/locality_masking." + fig_format,
                bbox_inches="tight",
                dpi=fig_png_dpi,
            )

    results = dict()

    params["stl_type"] = "D-SSR"
    results["f_ssr"], results["x_ssr"] = prox_linear(params)

    params["stl_type"] = "D-GMSR"
    results["f_dgmsr"], results["x_dgmsr"] = prox_linear(params)

    plot(params, results)
