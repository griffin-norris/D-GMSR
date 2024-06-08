import numpy as np
import typing as T

from .integration import rk4


def dVdt(
    V: np.ndarray,
    t: float,
    *args: T.Tuple[T.Any],
) -> np.ndarray:
    """
    ODE function to compute dVdt.
    V: Evaluation state V = [x, Phi_A, B_bar, C_bar, z_bar]
    t: Evaluation time
    u: Input at start of interval
    return: Derivative at current time and state dVdt
    """

    args = args[0]
    u_0, u_1, tt, params = args

    n_x = params["n_states"]
    n_u = params["n_controls"]

    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u
    i5 = i4 + n_x

    x = V[i0:i1]
    p = np.zeros(1)

    if params["inp_param"] == "ZOH":
        beta = 0.0
    elif params["inp_param"] == "FOH":
        beta = t / params["t_scp"]
    alpha = 1 - beta

    u = u_0 + beta * (u_1 - u_0)

    A_subs = params["A_func"](x, u)
    B_subs = params["B_func"](x, u)
    f_subs = params["f_func"](x, u)[:, 0]

    z_t = np.squeeze(f_subs) - np.matmul(A_subs, x) - np.matmul(B_subs, u)

    dVdt = np.zeros_like(V)
    dVdt[i0:i1] = f_subs.T
    dVdt[i1:i2] = np.matmul(A_subs, V[i1:i2].reshape((n_x, n_x))).reshape(-1)
    dVdt[i2:i3] = (
        np.matmul(A_subs, V[i2:i3].reshape((n_x, n_u))) + B_subs * alpha
    ).reshape(-1)
    dVdt[i3:i4] = (
        np.matmul(A_subs, V[i3:i4].reshape((n_x, n_u))) + B_subs * beta
    ).reshape(-1)
    dVdt[i4:i5] = np.matmul(A_subs, V[i4:i5]).reshape(-1) + z_t

    return dVdt


def calculate_discretization(
    X: np.ndarray,
    U: np.ndarray,
    params: T.Dict[str, T.Any],
) -> T.Dict[str, T.Any]:
    """
    Calculate discretization for given states, inputs and total time.
    X: Matrix of states for all time points
    U: Matrix of inputs for all time points
    return: The discretization matrices
    """

    n_x = params["n_states"]
    n_u = params["n_controls"]

    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u
    i5 = i4 + n_x

    V0 = np.zeros(i5)
    V0[i1:i2] = np.eye(n_x).reshape(-1)

    f_bar = np.zeros((n_x, params["K"] - 1))
    A_bar = np.zeros((n_x * n_x, params["K"] - 1))
    B_bar = np.zeros((n_x * n_u, params["K"] - 1))
    C_bar = np.zeros((n_x * n_u, params["K"] - 1))
    z_bar = np.zeros((n_x, params["K"] - 1))

    for k in range(params["K"] - 1):
        V0[i0:i1] = X[:, k]

        tt = k * params["t_scp"]
        V = rk4(
            dVdt,
            V0,
            params["t_scp"],
            params["rk4_steps_J"],
            U[:, k],
            U[:, k + 1],
            tt,
            params,
        )[-1, :]

        # flatten matrices in column-major (Fortran) order for cvxpy
        f_bar[:, k] = V[i0:i1]
        Phi = V[i1:i2].reshape((n_x, n_x))
        A_bar[:, k] = Phi.flatten(order="F")
        B_bar[:, k] = (V[i2:i3].reshape((n_x, n_u))).flatten(order="F")
        C_bar[:, k] = (V[i3:i4].reshape((n_x, n_u))).flatten(order="F")
        z_bar[:, k] = V[i4:i5]

    params["f_bar"] = f_bar
    params["A_bar"] = A_bar
    params["B_bar"] = B_bar
    params["C_bar"] = C_bar
    params["z_bar"] = z_bar

    return params
