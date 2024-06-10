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

    x0_and = np.array([-12., -9., -6., 3., 6.])
    alpha_and = 4.32

    x0_or = np.array([-8., -7., -6., -5.0, -4.95])
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
        
        h_and, c_i_w_i = f_func_smth(xk_and_2)[0,0], A_func_smth(xk_and_2)[0,:]
        xk_and_2 = xk_and_2 + alpha_and * c_i_w_i
        f_list_and_2.append(h_and)
        x_list_and_2.append(xk_and_2)
        
        h_or, d_i_w_i = gmsr_or(eps, p, weights, xk_or_1)
        xk_or_1 = xk_or_1 + alpha_or * d_i_w_i
        f_list_or_1.append(h_or)
        x_list_or_1.append(xk_or_1)
        
        h_or, d_i_w_i = f_func_smth(xk_or_2)[1,0], A_func_smth(xk_or_2)[1,:]
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
    # Results - And

    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 2]}, figsize=(8, 6))

    color_1 = iter(cm.rainbow(np.linspace(0, 1, N)))
    plt.figure()
    for i in range(K):
        c = next(color_1)
        axs[0,0].plot(x_list_and_2[:, N-1-i], c=c)
    axs[0,0].axhline(0, c='black', linestyle='dashed')
    axs[0,0].set_xlabel(r'iterations', fontsize=16)
    axs[0,0].set_ylabel('$f_i(x)$', fontsize=16)
    axs[0,0].set_title('D-SSR', fontsize=16)

    color_1 = iter(cm.rainbow(np.linspace(0, 1, K)))
    plt.figure()
    for i in range(K):
        c = next(color_1)
        axs[0,1].plot(x_list_and_1[:, K-1-i], c=c)
    axs[0,1].axhline(0, c='black', linestyle='dashed')
    axs[0,1].set_xlabel(r'iterations', fontsize=16)
    axs[0,1].set_ylabel('$f_i(x)$', fontsize=16)
    axs[0,1].set_title('D-GMSR', fontsize=16)

    axs[1,0].plot(f_list_and_2, c='blue')
    axs[1,0].axhline(0, c='black', linestyle='dashed')
    axs[1,0].set_xlabel(r'iterations', fontsize=16)
    axs[1,0].set_ylabel('$\\widetilde{\\min} ( y )$', fontsize=16)

    axs[1,1].plot(f_list_and_1, c='blue')
    axs[1,1].axhline(0, c='black', linestyle='dashed')
    axs[1,1].set_xlabel(r'iterations', fontsize=16)
    axs[1,1].set_ylabel('$^{\\wedge} h_{p, w}^{\\epsilon}( y )$', fontsize=16)

    if save_fig: fig.savefig('figs/and_ssr_dgmsr.' + fig_format, bbox_inches='tight', dpi=fig_png_dpi)  

    # ---------------------------------------------------------------
    # Results - Or

    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 2]}, figsize=(8, 6))
 
    color_1 = iter(cm.rainbow(np.linspace(0, 1, K)))
    plt.figure()
    for i in range(K):
        c = next(color_1)
        axs[0,0].plot(x_list_or_2[:, K-1-i], c=c)
    axs[0,0].axhline(0, c='black', linestyle='dashed')
    axs[0,0].set_xlabel(r'iterations', fontsize=16)
    axs[0,0].set_ylabel('$f_i(x)$', fontsize=16)
    axs[0,0].set_title('D-SSR', fontsize=16)

    color_1 = iter(cm.rainbow(np.linspace(0, 1, K)))
    plt.figure()
    for i in range(K):
        c = next(color_1)
        axs[0,1].plot(x_list_or_1[:, N-1-i], c=c)
    axs[0,1].axhline(0, c='black', linestyle='dashed')
    axs[0,1].set_xlabel(r'iterations', fontsize=16)
    axs[0,1].set_ylabel('$f_i(x)$', fontsize=16)
    axs[0,1].set_title('D-GMSR', fontsize=16)

    axs[1,0].plot(f_list_or_2, c='blue')
    axs[1,0].axhline(0, c='black', linestyle='dashed')
    axs[1,0].set_xlabel(r'iterations', fontsize=16)
    axs[1,0].set_ylabel('$$\\widetilde{\\max} ( y )$$', fontsize=14)

    axs[1,1].plot(f_list_or_1, c='blue')
    axs[1,1].axhline(0, c='black', linestyle='dashed')
    axs[1,1].set_xlabel(r'iterations', fontsize=16)
    axs[1,1].set_ylabel('$^{\\vee} h_{p, w}^{\\epsilon}(y)$', fontsize=16)

    if save_fig: fig.savefig('figs/or_ssr_dgmsr.' + fig_format, bbox_inches='tight', dpi=fig_png_dpi)  
