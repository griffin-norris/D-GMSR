import numpy as np
import sympy as sp
import typing as T


def rl_dynamics(params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:

    f = sp.zeros(params["n_states"], 1)

    x = sp.Matrix(sp.symbols("m rx rz vx vz theta omega delta", real=True))
    u = sp.Matrix(sp.symbols("T1 T2 T3 delta_dot", real=True))

    alpha_m = 1 / (params["I_sp"] * params["g_0"])
    rot_tr = sp.Matrix([[sp.cos(x[5, 0] + x[7, 0])], [-sp.sin(x[5, 0] + x[7, 0])]])
    R_IB = sp.Matrix(
        [[sp.cos(x[5, 0]), sp.sin(x[5, 0])], [-sp.sin(x[5, 0]), sp.cos(x[5, 0])]]
    )

    g_I = sp.Matrix(np.array([-params["g_0"], 0.0]))
    J_B = x[0, 0] * ((params["l_r"] ** 2) / 4 + (params["l_h"] ** 2) / 12)

    v_norm = (x[3, 0] ** 2 + x[4, 0] ** 2 + 1e-8) ** (0.5)
    A_B = (
        -0.5
        * params["rho_air"]
        * params["S_area"]
        * v_norm
        * (params["C_aero"] @ (R_IB.T @ x[3:5, 0]))
    )
    A_I = R_IB @ A_B

    f[0, 0] = -alpha_m * (u[0, 0] + u[1, 0] + u[2, 0])
    f[1:3, 0] = x[3:5, 0]
    f[3:5, 0] = g_I + ((rot_tr * (u[0, 0] + u[1, 0] + u[2, 0])) + A_I) / x[0, 0]
    f[5, 0] = x[6, 0]
    f[6, 0] = (
        params["l_cm"] * (u[0, 0] + u[1, 0] + u[2, 0]) * -sp.sin(x[7, 0])
        - params["l_cp"] * A_B[1, 0]
    ) / J_B
    f[7, 0] = u[3, 0]

    f = sp.simplify(f)
    A = sp.simplify(f.jacobian(x))
    B = sp.simplify(f.jacobian(u))

    params["f_func"] = sp.lambdify((x, u), f, "numpy")
    params["A_func"] = sp.lambdify((x, u), A, "numpy")
    params["B_func"] = sp.lambdify((x, u), B, "numpy")

    return params
