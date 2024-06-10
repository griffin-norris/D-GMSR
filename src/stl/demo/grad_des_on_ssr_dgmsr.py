def grad_des_on_ssr_dgmsr():

    # ---------------------------------------------------------------
    # D-SSR D-GMSR Parameters

    K = 5
    k = 25
    p = 1
    eps = 1e-8
    weights = np.ones(K)

    f_func_smth, A_func_smth = ssr_sympy(k=k, K=K)

    # ---------------------------------------------------------------
    # Gradient Descent Parameters

    ITE = 20

    x0_and = np.array([-12.0, -9.0, -6.0, 3.0, 6.0])
    alpha_and = 4.32

    x0_or = np.array([-8.0, -7.0, -6.0, -5.0, -4.95])
    alpha_or = 2.1

    # ---------------------------------------------------------------
    # Gradient Descent

    xk_and_1 = x0_and.copy()
    x_list_and_1 = [xk_and_1]
    f_list_and_1 = []

    xk_and_2 = x0_and.copy()
    x_list_and_2 = [xk_and_2]
    f_list_and_2 = []

    xk_or_1 = x0_or.copy()
    x_list_or_1 = [xk_or_1]
    f_list_or_1 = []

    xk_or_2 = x0_or.copy()
    x_list_or_2 = [xk_or_2]
    f_list_or_2 = []

    for i in range(ITE):
        h_and, c_i_w_i = gmsr_and(eps, p, weights, xk_and_1)
        xk_and_1 = xk_and_1 + alpha_and * c_i_w_i
        f_list_and_1.append(h_and)
        x_list_and_1.append(xk_and_1)

        # h_and, c_i_w_i = gmsr_and(eps, 32, weights, xk_and_2)
        h_and, c_i_w_i = f_func_smth(xk_and_2)[0, 0], A_func_smth(xk_and_2)[0, :]

        xk_and_2 = xk_and_2 + alpha_and * c_i_w_i
        f_list_and_2.append(h_and)
        x_list_and_2.append(xk_and_2)

        h_or, d_i_w_i = gmsr_or(eps, p, weights, xk_or_1)
        xk_or_1 = xk_or_1 + alpha_or * d_i_w_i
        f_list_or_1.append(h_or)
        x_list_or_1.append(xk_or_1)

        h_or, d_i_w_i = f_func_smth(xk_or_2)[1, 0], A_func_smth(xk_or_2)[1, :]
        xk_or_2 = xk_or_2 + alpha_or * d_i_w_i
        f_list_or_2.append(h_or)
        x_list_or_2.append(xk_or_2)

    x_list_and_1 = np.vstack(x_list_and_1)
    f_list_and_1 = np.vstack(f_list_and_1)

    x_list_and_2 = np.vstack(x_list_and_2)
    f_list_and_2 = np.vstack(f_list_and_2)

    x_list_or_1 = np.vstack(x_list_or_1)
    f_list_or_1 = np.vstack(f_list_or_1)

    x_list_or_2 = np.vstack(x_list_or_2)
    f_list_or_2 = np.vstack(f_list_or_2)

    # ---------------------------------------------------------------
    # Results - D-SSR

    fig, axs = plt.subplots(2, 2, gridspec_kw={"height_ratios": [3, 2]}, figsize=(8, 6))

    color_1 = iter(cm.rainbow(np.linspace(0, 1, K)))
    plt.figure()
    for i in range(K):
        c = next(color_1)
        axs[0, 0].plot(x_list_and_2[:, K - 1 - i], c=c)
    axs[0, 0].axhline(0, c="black", linestyle="dashed")
    axs[0, 0].set_xlabel(r"iterations", fontsize=16)
    axs[0, 0].set_ylabel("$f_i(x_k)$", fontsize=16)
    axs[0, 0].set_title("Conjuction - Always", fontsize=16)

    axs[1, 0].plot(f_list_and_2, c="blue")
    axs[1, 0].axhline(0, c="black", linestyle="dashed")
    axs[1, 0].set_xlabel(r"iterations", fontsize=16)
    axs[1, 0].set_ylabel(
        "$\\tilde{\\rho}^{\\varphi_1 \\wedge \\varphi_2 \\wedge \\dots \\wedge \\varphi_5} (x,k)$",
        fontsize=16,
    )

    color_1 = iter(cm.rainbow(np.linspace(0, 1, K)))
    plt.figure()
    for i in range(K):
        c = next(color_1)
        axs[0, 1].plot(x_list_or_2[:, K - 1 - i], c=c)
    axs[0, 1].axhline(0, c="black", linestyle="dashed")
    axs[0, 1].set_xlabel(r"iterations", fontsize=16)
    axs[0, 1].set_ylabel("$f_i(x_k)$", fontsize=16)
    axs[0, 1].set_title("Disjunction - Eventually", fontsize=16)

    axs[1, 1].plot(f_list_or_2, c="blue")
    axs[1, 1].axhline(0, c="black", linestyle="dashed")
    axs[1, 1].set_xlabel(r"iterations", fontsize=16)
    axs[1, 1].set_ylabel(
        "$\\tilde{\\rho}^{\\varphi_1 \\vee \\varphi_2 \\vee \\dots \\vee \\varphi_5} (x,k)$",
        fontsize=16,
    )

    if save_fig:
        fig.savefig("figs/dssr." + fig_format, bbox_inches="tight", dpi=fig_png_dpi)

    # ---------------------------------------------------------------
    # Results - D-GMSR

    fig, axs = plt.subplots(2, 2, gridspec_kw={"height_ratios": [3, 2]}, figsize=(8, 6))

    color_1 = iter(cm.rainbow(np.linspace(0, 1, K)))
    plt.figure()
    for i in range(K):
        c = next(color_1)
        axs[0, 0].plot(x_list_and_1[:, K - 1 - i], c=c)
    axs[0, 0].axhline(0, c="black", linestyle="dashed")
    axs[0, 0].set_xlabel(r"iterations", fontsize=16)
    axs[0, 0].set_ylabel("$f_i(x_k)$", fontsize=16)
    axs[0, 0].set_title("Conjuction - Always", fontsize=16)

    axs[1, 0].plot(f_list_and_1, c="blue")
    axs[1, 0].axhline(0, c="black", linestyle="dashed")
    axs[1, 0].set_xlabel(r"iterations", fontsize=16)
    axs[1, 0].set_ylabel(
        "$\\Gamma^{\\varphi_1 \\wedge \\varphi_2 \\wedge \\dots \\wedge \\varphi_5}_{\\bm{\\epsilon}, \\bm{p}, \\bm{w}} (x,k)$",
        fontsize=16,
    )

    color_1 = iter(cm.rainbow(np.linspace(0, 1, K)))
    plt.figure()
    for i in range(K):
        c = next(color_1)
        axs[0, 1].plot(x_list_or_1[:, K - 1 - i], c=c)
    axs[0, 1].axhline(0, c="black", linestyle="dashed")
    axs[0, 1].set_xlabel(r"iterations", fontsize=16)
    axs[0, 1].set_ylabel("$f_i(x_k)$", fontsize=16)
    axs[0, 1].set_title("Disjunction - Eventually", fontsize=16)

    axs[1, 1].plot(f_list_or_1, c="blue")
    axs[1, 1].axhline(0, c="black", linestyle="dashed")
    axs[1, 1].set_xlabel(r"iterations", fontsize=16)
    axs[1, 1].set_ylabel(
        "$\\Gamma^{\\varphi_1 \\vee \\varphi_2 \\vee \\dots \\vee \\varphi_5}_{\\bm{\\epsilon}, \\bm{p}, \\bm{w}} (x,k)$",
        fontsize=16,
    )

    if save_fig:
        fig.savefig("figs/dgmsr." + fig_format, bbox_inches="tight", dpi=fig_png_dpi)
