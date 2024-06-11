import numpy as np
import cvxpy as cp
import typing as T

import src.stl as stl


def f_wp(xk: np.ndarray, dx: np.ndarray, p_w: np.ndarray, npy: bool) -> T.Tuple[T.Any]:

    dist_c = np.linalg.norm(xk[:3, :] - p_w[:3, :], axis=0)
    dist = p_w[3, 0] - dist_c
    direction = -(xk[:3, :] - p_w[:3, :]) / (dist_c + 1e-8)

    f_0 = dist
    if not (npy):
        l_k = f_0 + cp.sum(cp.multiply(direction, dx[:3, :]), axis=0)
    else:
        l_k = f_0 + (direction * dx[:3, :]).sum(axis=0)

    return f_0, l_k


def f_spd(
    xk: np.ndarray, dx: np.ndarray, spd_lim: np.ndarray, npy: bool
) -> T.Tuple[T.Any]:

    speed = np.linalg.norm(xk[3:6, :], axis=0)
    dist = spd_lim - speed
    direction = -(xk[3:6, :]) / (speed + 1e-8)

    f_0 = dist
    if not (npy):
        l_k = f_0 + cp.sum(cp.multiply(direction, dx[3:6, :]), axis=0)
    else:
        l_k = f_0 + (direction * dx[3:6, :]).sum(axis=0)

    return f_0, l_k


def f_until(
    xk: np.ndarray, dxk: np.ndarray, params: T.Dict, npy: bool = True
) -> T.Tuple[T.Any]:

    # -----------------------------------------------------------

    f_0_wp, l_k_wp = f_wp(xk=xk, dx=dxk, p_w=params["p_w"], npy=npy)

    # -----------------------------------------------------------

    f_0_spd, l_k_spd = f_spd(
        xk=xk, dx=dxk, spd_lim=params["vehicle_v_max_evnt"], npy=npy
    )

    # -----------------------------------------------------------

    f_0_wp_c = []
    f_0_spd_c = []
    l_k_wp_c = []
    l_k_spd_c = []
    for k in range(params["K"] - params["t_w"] + 1):

        wait_wp = f_0_wp[k : k + params["t_w"]]
        wait_spd = f_0_spd[k : k + params["t_w"]]

        f_0_wp_c_k, grad_wp_c_k = stl.gmsr.gmsr_and(
            params["stl_eps"], params["stl_p"], params["stl_w_c"], wait_wp
        )

        f_0_spd_c_k, grad_spd_c_k = stl.gmsr.gmsr_and(
            params["stl_eps"], params["stl_p"], params["stl_w_c"], wait_spd
        )

        l_k_wp_c_k = f_0_wp_c_k
        l_k_spd_c_k = f_0_spd_c_k
        for i in range(params["t_w"]):
            l_k_wp_c_k += grad_wp_c_k[i] * (l_k_wp[k + i] - f_0_wp[k + i])
            l_k_spd_c_k += grad_spd_c_k[i] * (l_k_spd[k + i] - f_0_spd[k + i])

        f_0_wp_c.append(f_0_wp_c_k)
        f_0_spd_c.append(f_0_spd_c_k)
        l_k_wp_c.append(l_k_wp_c_k)
        l_k_spd_c.append(l_k_spd_c_k)

    f_0_wp_c = np.vstack(f_0_wp_c)[:, 0]
    f_0_spd_c = np.vstack(f_0_spd_c)[:, 0]
    if npy:
        l_k_wp_c = np.vstack(l_k_wp_c)[:, 0]
        l_k_spd_c = np.vstack(l_k_spd_c)[:, 0]
    else:
        l_k_wp_c = cp.vstack(l_k_wp_c)[:, 0]
        l_k_spd_c = cp.vstack(l_k_spd_c)[:, 0]

    f_0_wp = f_0_wp_c
    f_0_spd = f_0_spd_c

    l_k_wp = l_k_wp_c
    l_k_spd = l_k_spd_c

    # ---------------------------------------------------------------

    f_0_until, phi_1_grad, phi_2_grad = stl.until.UNTIL(
        params["stl_eps"],
        params["stl_p"],
        params["stl_w_phi_1"],
        params["stl_w_phi_2"],
        params["stl_w_phi_12"],
        f_0_spd,
        f_0_wp,
    )

    # ---------------------------------------------------------------

    if npy:
        l_k_until = f_0_until + np.sum(
            phi_2_grad * (l_k_wp - f_0_wp) + phi_1_grad * (l_k_spd - f_0_spd)
        )
    else:
        l_k_until = f_0_until + cp.sum(
            phi_2_grad * (l_k_wp - f_0_wp) + phi_1_grad * (l_k_spd - f_0_spd)
        )

    return -f_0_until, -l_k_until


def qf_cost_fcn(
    X: np.ndarray,
    U: np.ndarray,
    X_last: np.ndarray,
    U_last: np.ndarray,
    params: T.Dict[str, T.Any],
    npy: bool,
) -> T.Tuple[T.Any]:

    _, lin_until_cost = f_until(X_last, X - X_last, params, npy=npy)
    vehicle_cost = params["w_obj_stl"] * lin_until_cost

    return vehicle_cost
