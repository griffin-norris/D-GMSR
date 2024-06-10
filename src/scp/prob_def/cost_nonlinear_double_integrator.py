import numpy as np
import cvxpy as cp
import typing as T

from src.stl.until import UNTIL


def ndi_f_wp(
    xk: np.ndarray, dx: np.ndarray, p_w: np.ndarray, npy: bool
) -> T.Tuple[T.Any]:

    dist_c = np.linalg.norm(xk[:2, :] - p_w[:2, :], axis=0)
    dist = p_w[2, 0] - dist_c
    direction = -(xk[:2, :] - p_w[:2, :]) / (dist_c + 1e-8)

    f_0 = dist
    if not (npy):
        l_k = f_0 + cp.sum(cp.multiply(direction, dx[:2, :]), axis=0)
    else:
        l_k = f_0 + (direction * dx[:2, :]).sum(axis=0)

    return f_0, l_k


def ndi_f_spd(
    xk: np.ndarray, dx: np.ndarray, spd_lim: np.ndarray, npy: bool
) -> T.Tuple[T.Any]:

    speed = np.linalg.norm(xk[2:4, :], axis=0)
    dist = spd_lim - speed
    direction = -(xk[2:4, :]) / (speed + 1e-8)

    f_0 = dist
    if not (npy):
        l_k = f_0 + cp.sum(cp.multiply(direction, dx[2:4, :]), axis=0)
    else:
        l_k = f_0 + (direction * dx[2:4, :]).sum(axis=0)

    return f_0, l_k


def ndi_f_until(
    xk: np.ndarray, dxk: np.ndarray, params: T.Dict, npy: bool = True
) -> T.Tuple[T.Any]:

    # ---------------------------------------------------
    f_0_wp, l_k_wp = ndi_f_wp(xk=xk, dx=dxk, p_w=params["p_w"], npy=npy)
    # ---------------------------------------------------
    f_0_spd, l_k_spd = ndi_f_spd(
        xk=xk, dx=dxk, spd_lim=params["vehicle_v_max_evnt"], npy=npy
    )

    # ---------------------------------------------------
    f_0_until, phi_1_grad, phi_2_grad = UNTIL(
        params["stl_eps"],
        params["stl_p"],
        params["stl_w_phi_1"],
        params["stl_w_phi_2"],
        params["stl_w_phi_12"],
        f_0_spd,
        f_0_wp,
    )
    # ---------------------------------------------------

    if npy:
        l_k_until = f_0_until + np.sum(
            phi_2_grad * (l_k_wp - f_0_wp) + phi_1_grad * (l_k_spd - f_0_spd)
        )
    else:
        l_k_until = f_0_until + cp.sum(
            phi_2_grad * (l_k_wp - f_0_wp) + phi_1_grad * (l_k_spd - f_0_spd)
        )

    return -f_0_until, -l_k_until


def ndi_cost_fcn(
    X: np.ndarray,
    U: np.ndarray,
    X_last: np.ndarray,
    U_last: np.ndarray,
    params: T.Dict[str, T.Any],
    npy: bool,
) -> T.Tuple[T.Any]:

    _, lin_until_cost = ndi_f_until(X_last, X - X_last, params, npy=npy)
    vehicle_cost = params["w_obj_stl"] * lin_until_cost

    return vehicle_cost
