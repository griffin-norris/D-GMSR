import numpy as np
import cvxpy as cp
import typing as T

from src.stl.until import UNTIL
from src.stl.gmsr import gmsr_or


def rl_f_norm(
    xk: np.ndarray, dx: np.ndarray, spd_bd: float, npy: bool
) -> T.Tuple[T.Any]:

    speed = np.linalg.norm(xk, axis=0)
    dist = spd_bd - speed
    direction = -(xk) / (speed + 1e-8)

    f_0 = dist
    if not (npy):
        l_k = f_0 + cp.sum(cp.multiply(direction, dx), axis=0)
    else:
        l_k = f_0 + (direction * dx).sum(axis=0)

    return f_0, l_k


def rl_f_scl(xk: np.ndarray, dx: np.ndarray, lw_bd: float, npy: bool) -> T.Tuple[T.Any]:

    f_0 = xk - lw_bd
    l_k = f_0 + dx

    return f_0, l_k


def rl_f_gs(xk: np.ndarray, dx: np.ndarray, gs_ang: float, npy: bool) -> T.Tuple[T.Any]:

    f_0 = xk[1] - xk[0] * np.tan(-gs_ang)
    l_k = f_0 + dx[1] + np.tan(gs_ang) * dx[0]

    return f_0, l_k


def rl_f_stc(
    f_0_tr: T.Any,
    l_k_tr: T.Any,
    f_0_spd: T.Any,
    l_k_spd: T.Any,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:

    f_0_stc = []
    l_k_stc = []
    for k in range(params["K"]):

        f_0_stc_k, phi_grad_k = gmsr_or(
            params["stl_eps"],
            params["stl_p"],
            params["stl_w_phi"],
            np.array([f_0_tr[k], f_0_spd[k]]),
        )
        f_0_stc.append(f_0_stc_k)
        g1 = phi_grad_k[0] * (l_k_tr[k] - f_0_tr[k])
        g2 = phi_grad_k[1] * (l_k_spd[k] - f_0_spd[k])
        l_k_stc.append(f_0_stc_k + g1 + g2)

    f_0_stc = np.vstack(f_0_stc)

    # ---------------------------------------------------

    if npy:
        l_k_stc = np.vstack(l_k_stc)
    else:
        l_k_stc = cp.vstack(l_k_stc)

    return -f_0_stc, -l_k_stc


def f_spd_trig_tr_E1_L(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 100
    w2 = 10
    # ------------------------------------------------------------
    f_0_tr, l_k_tr = rl_f_scl(xk=uk[1, :], dx=du[1, :], lw_bd=params["T_min"], npy=npy)

    f_0_tr, l_k_tr = w1 * f_0_tr, w1 * l_k_tr
    # ------------------------------------------------------------
    f_0_spd, l_k_spd = rl_f_norm(
        xk=xk[3:5, :], dx=dx[3:5, :], spd_bd=params["spd_trig"], npy=npy
    )

    f_0_spd, l_k_spd = w2 * f_0_spd, w2 * l_k_spd
    # ------------------------------------------------------------

    return rl_f_stc(f_0_tr, l_k_tr, f_0_spd, l_k_spd, params, npy)


def f_spd_trig_tr_E1_U(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 100
    w2 = 10
    # ------------------------------------------------------------
    f_0_tr, l_k_tr = rl_f_scl(xk=uk[1, :], dx=du[1, :], lw_bd=0.0, npy=npy)

    f_0_tr, l_k_tr = -w1 * f_0_tr, -w1 * l_k_tr
    # ------------------------------------------------------------
    f_0_spd, l_k_spd = rl_f_norm(
        xk=xk[3:5, :], dx=dx[3:5, :], spd_bd=params["spd_trig"], npy=npy
    )

    f_0_spd, l_k_spd = -w2 * f_0_spd, -w2 * l_k_spd
    # ------------------------------------------------------------

    return rl_f_stc(f_0_tr, l_k_tr, f_0_spd, l_k_spd, params, npy)


def f_spd_trig_tr_E2_L(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 100
    w2 = 10
    # ------------------------------------------------------------
    f_0_tr, l_k_tr = rl_f_scl(xk=uk[2, :], dx=du[2, :], lw_bd=params["T_min"], npy=npy)

    f_0_tr, l_k_tr = w1 * f_0_tr, w1 * l_k_tr
    # ------------------------------------------------------------
    f_0_spd, l_k_spd = rl_f_norm(
        xk=xk[3:5, :], dx=dx[3:5, :], spd_bd=params["spd_trig"], npy=npy
    )

    f_0_spd, l_k_spd = w2 * f_0_spd, w2 * l_k_spd
    # ------------------------------------------------------------

    return rl_f_stc(f_0_tr, l_k_tr, f_0_spd, l_k_spd, params, npy)


def f_spd_trig_tr_E2_U(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 100
    w2 = 10
    # ------------------------------------------------------------
    f_0_tr, l_k_tr = rl_f_scl(xk=uk[2, :], dx=du[2, :], lw_bd=0.0, npy=npy)

    f_0_tr, l_k_tr = -w1 * f_0_tr, -w1 * l_k_tr
    # ------------------------------------------------------------
    f_0_spd, l_k_spd = rl_f_norm(
        xk=xk[3:5, :], dx=dx[3:5, :], spd_bd=params["spd_trig"], npy=npy
    )

    f_0_spd, l_k_spd = -w2 * f_0_spd, -w2 * l_k_spd
    # ------------------------------------------------------------

    return rl_f_stc(f_0_tr, l_k_tr, f_0_spd, l_k_spd, params, npy)


def f_alt_trig_spd(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 1000
    w2 = 100
    # -------------------------------------------------------------------
    f_0_alt, l_k_alt = rl_f_scl(
        xk=xk[1, :], dx=dx[1, :], lw_bd=params["alt_trig"], npy=npy
    )

    f_0_alt, l_k_alt = w1 * f_0_alt, w1 * l_k_alt
    # -------------------------------------------------------------------
    f_0_spd, l_k_spd = rl_f_norm(
        xk=xk[3:5, :], dx=dx[3:5, :], spd_bd=params["spd_trig_aft"], npy=npy
    )

    f_0_spd, l_k_spd = w2 * f_0_spd, w2 * l_k_spd
    # -------------------------------------------------------------------

    return rl_f_stc(f_0_alt, l_k_alt, f_0_spd, l_k_spd, params, npy)


def f_spd_trig_spd(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 100
    w2 = 100
    f_0_stc = []
    l_k_stc = []
    for k in range(params["K"] - 1):

        f_0_k0 = params["spd_trig"] - np.linalg.norm(xk[3:5, k])
        f_0_k1 = params["spd_trig"] - np.linalg.norm(xk[3:5, k + 1])

        g_0_k0 = -(xk[3:5, k] / (1e-8 + np.linalg.norm(xk[3:5, k])))
        g_0_k1 = -(xk[3:5, k + 1] / (1e-8 + np.linalg.norm(xk[3:5, k + 1])))

        l_0_k0 = f_0_k0 + g_0_k0 @ dx[3:5, k]
        l_0_k1 = f_0_k1 + g_0_k1 @ dx[3:5, k]

        f_0_k0 = -w1 * f_0_k0
        l_0_k0 = -w1 * l_0_k0

        f_0_k1 = w2 * f_0_k1
        l_0_k1 = w2 * l_0_k1

        f_0_stc_k, phi_grad_k = gmsr_or(
            params["stl_eps"],
            params["stl_p"],
            params["stl_w_phi"],
            np.array([f_0_k0, f_0_k1]),
        )

        f_0_stc.append(f_0_stc_k)
        g1 = phi_grad_k[0] * (l_0_k0 - f_0_k0)
        g2 = phi_grad_k[1] * (l_0_k1 - f_0_k1)
        l_k_stc.append(f_0_stc_k + g1 + g2)

    f_0_stc = np.vstack(f_0_stc)

    # -------------------------------------------------------------------

    if npy:
        l_k_stc = np.vstack(l_k_stc)
    else:
        l_k_stc = cp.vstack(l_k_stc)

    return -f_0_stc, -l_k_stc


def f_alt_trig_theta_U(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 1000
    w2 = 10
    # -------------------------------------------------------------------
    f_0_alt, l_k_alt = rl_f_scl(
        xk=xk[1, :], dx=dx[1, :], lw_bd=params["alt_trig"], npy=npy
    )

    f_0_alt, l_k_alt = w1 * f_0_alt, w1 * l_k_alt
    # -------------------------------------------------------------------
    f_0_theta, l_k_theta = rl_f_scl(
        xk=xk[5, :], dx=dx[5, :], lw_bd=params["theta_trig_aft"], npy=npy
    )
    f_0_theta, l_k_theta = -w2 * f_0_theta, -w2 * l_k_theta
    # -------------------------------------------------------------------

    return rl_f_stc(f_0_alt, l_k_alt, f_0_theta, l_k_theta, params, npy)


def f_alt_trig_theta_L(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 1000
    w2 = 10
    # -------------------------------------------------------------------
    f_0_alt, l_k_alt = rl_f_scl(
        xk=xk[1, :], dx=dx[1, :], lw_bd=params["alt_trig"], npy=npy
    )
    f_0_alt, l_k_alt = w1 * f_0_alt, w1 * l_k_alt
    # -------------------------------------------------------------------
    f_0_theta, l_k_theta = rl_f_scl(
        xk=xk[5, :], dx=dx[5, :], lw_bd=-params["theta_trig_aft"], npy=npy
    )
    f_0_theta, l_k_theta = w2 * f_0_theta, w2 * l_k_theta
    # -------------------------------------------------------------------

    return rl_f_stc(f_0_alt, l_k_alt, f_0_theta, l_k_theta, params, npy)


def f_alt_trig_omega_U(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 1000
    w2 = 10
    # -------------------------------------------------------------------
    f_0_alt, l_k_alt = rl_f_scl(
        xk=xk[1, :], dx=dx[1, :], lw_bd=params["alt_trig"], npy=npy
    )
    f_0_alt, l_k_alt = w1 * f_0_alt, w1 * l_k_alt
    # -------------------------------------------------------------------
    f_0_omega, l_k_omega = rl_f_scl(
        xk=xk[6, :], dx=dx[6, :], lw_bd=params["omega_trig_aft"], npy=npy
    )

    f_0_omega, l_k_omega = -w2 * f_0_omega, -w2 * l_k_omega
    # -------------------------------------------------------------------

    return rl_f_stc(f_0_alt, l_k_alt, f_0_omega, l_k_omega, params, npy)


def f_alt_trig_omega_L(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 1000
    w2 = 10
    # -------------------------------------------------------------------
    f_0_alt, l_k_alt = rl_f_scl(
        xk=xk[1, :], dx=dx[1, :], lw_bd=params["alt_trig"], npy=npy
    )

    f_0_alt, l_k_alt = w1 * f_0_alt, w1 * l_k_alt
    # -------------------------------------------------------------------
    f_0_omega, l_k_omega = rl_f_scl(
        xk=xk[6, :], dx=dx[6, :], lw_bd=-params["omega_trig_aft"], npy=npy
    )
    f_0_omega, l_k_omega = w2 * f_0_omega, w2 * l_k_omega
    # -------------------------------------------------------------------

    return rl_f_stc(f_0_alt, l_k_alt, f_0_omega, l_k_omega, params, npy)


def f_alt_trig_delta_U(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 1000
    w2 = 10
    # -------------------------------------------------------------------
    f_0_alt, l_k_alt = rl_f_scl(
        xk=xk[1, :], dx=dx[1, :], lw_bd=params["alt_trig"], npy=npy
    )
    f_0_alt, l_k_alt = w1 * f_0_alt, w1 * l_k_alt
    # -------------------------------------------------------------------
    f_0_delta, l_k_delta = rl_f_scl(
        xk=xk[7, :], dx=dx[7, :], lw_bd=params["delta_trig_aft"], npy=npy
    )
    f_0_delta, l_k_delta = -w2 * f_0_delta, -w2 * l_k_delta
    # -------------------------------------------------------------------

    return rl_f_stc(f_0_alt, l_k_alt, f_0_delta, l_k_delta, params, npy)


def f_alt_trig_delta_L(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 1000
    w2 = 10
    # -------------------------------------------------------------------
    f_0_alt, l_k_alt = rl_f_scl(
        xk=xk[1, :], dx=dx[1, :], lw_bd=params["alt_trig"], npy=npy
    )
    f_0_alt, l_k_alt = w1 * f_0_alt, w1 * l_k_alt
    # -------------------------------------------------------------------
    f_0_delta, l_k_delta = rl_f_scl(
        xk=xk[7, :], dx=dx[7, :], lw_bd=-params["delta_trig_aft"], npy=npy
    )
    f_0_delta, l_k_delta = w2 * f_0_delta, w2 * l_k_delta
    # -------------------------------------------------------------------

    return rl_f_stc(f_0_alt, l_k_alt, f_0_delta, l_k_delta, params, npy)


def f_alt_trig_gs_U(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 1000
    w2 = 10
    # -------------------------------------------------------------------
    f_0_alt, l_k_alt = rl_f_scl(
        xk=xk[1, :], dx=dx[1, :], lw_bd=params["alt_trig"], npy=npy
    )

    f_0_alt, l_k_alt = w1 * f_0_alt, w1 * l_k_alt
    # -------------------------------------------------------------------
    f_0_gs, l_k_gs = rl_f_gs(
        xk=xk[1:3, :], dx=dx[1:3, :], gs_ang=params["gs_trig_aft"], npy=npy
    )

    f_0_gs, l_k_gs = w2 * f_0_gs, w2 * l_k_gs
    # -------------------------------------------------------------------

    return rl_f_stc(f_0_alt, l_k_alt, f_0_gs, l_k_gs, params, npy)


def f_alt_trig_gs_L(
    xk: np.ndarray,
    uk: np.ndarray,
    dx: np.ndarray,
    du: np.ndarray,
    params: T.Dict,
    npy: bool,
) -> T.Tuple[T.Any]:
    w1 = 1000
    w2 = 10
    # -------------------------------------------------------------------
    f_0_alt, l_k_alt = rl_f_scl(
        xk=xk[1, :], dx=dx[1, :], lw_bd=params["alt_trig"], npy=npy
    )

    f_0_alt, l_k_alt = w1 * f_0_alt, w1 * l_k_alt
    # -------------------------------------------------------------------
    f_0_gs, l_k_gs = rl_f_gs(
        xk=xk[1:3, :], dx=dx[1:3, :], gs_ang=-params["gs_trig_aft"], npy=npy
    )

    f_0_gs, l_k_gs = -w2 * f_0_gs, -w2 * l_k_gs

    # -------------------------------------------------------------------

    return rl_f_stc(f_0_alt, l_k_alt, f_0_gs, l_k_gs, params, npy)


def rl_cost_fcn(
    X: np.ndarray,
    U: np.ndarray,
    X_last: np.ndarray,
    U_last: np.ndarray,
    params: T.Dict[str, T.Any],
    npy: bool,
) -> T.Tuple[T.Any]:
    # ---------------------------------------------------------------

    _, lin_stc_E1_L = f_spd_trig_tr_E1_L(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_E1_L = np.sum(np.maximum(0.0, lin_stc_E1_L))
    else:
        lin_stc_E1_L = cp.sum(cp.maximum(0.0, lin_stc_E1_L))

    # ---------------------------------------------------------------

    _, lin_stc_E1_U = f_spd_trig_tr_E1_U(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_E1_U = np.sum(np.maximum(0.0, lin_stc_E1_U))
    else:
        lin_stc_E1_U = cp.sum(cp.maximum(0.0, lin_stc_E1_U))

    # ---------------------------------------------------------------

    _, lin_stc_E2_L = f_spd_trig_tr_E2_L(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_E2_L = np.sum(np.maximum(0.0, lin_stc_E2_L))
    else:
        lin_stc_E2_L = cp.sum(cp.maximum(0.0, lin_stc_E2_L))

    # ---------------------------------------------------------------

    _, lin_stc_E2_U = f_spd_trig_tr_E2_U(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_E2_U = np.sum(np.maximum(0.0, lin_stc_E2_U))
    else:
        lin_stc_E2_U = cp.sum(cp.maximum(0.0, lin_stc_E2_U))

    # ---------------------------------------------------------------

    _, lin_stc_spd_spd = f_spd_trig_spd(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_spd_spd = np.sum(np.maximum(0.0, lin_stc_spd_spd))
    else:
        lin_stc_spd_spd = cp.sum(cp.maximum(0.0, lin_stc_spd_spd))

    # ---------------------------------------------------------------

    _, lin_stc_alt_spd = f_alt_trig_spd(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_alt_spd = np.sum(np.maximum(0.0, lin_stc_alt_spd))
    else:
        lin_stc_alt_spd = cp.sum(cp.maximum(0.0, lin_stc_alt_spd))

    # ---------------------------------------------------------------

    _, lin_stc_alt_theta_U = f_alt_trig_theta_U(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_alt_theta_U = np.sum(np.maximum(0.0, lin_stc_alt_theta_U))
    else:
        lin_stc_alt_theta_U = cp.sum(cp.maximum(0.0, lin_stc_alt_theta_U))

    # ---------------------------------------------------------------

    _, lin_stc_alt_theta_L = f_alt_trig_theta_L(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_alt_theta_L = np.sum(np.maximum(0.0, lin_stc_alt_theta_L))
    else:
        lin_stc_alt_theta_L = cp.sum(cp.maximum(0.0, lin_stc_alt_theta_L))

    # ---------------------------------------------------------------

    _, lin_stc_alt_omega_U = f_alt_trig_omega_U(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_alt_omega_U = np.sum(np.maximum(0.0, lin_stc_alt_omega_U))
    else:
        lin_stc_alt_omega_U = cp.sum(cp.maximum(0.0, lin_stc_alt_omega_U))

    # ---------------------------------------------------------------

    _, lin_stc_alt_omega_L = f_alt_trig_omega_L(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_alt_omega_L = np.sum(np.maximum(0.0, lin_stc_alt_omega_L))
    else:
        lin_stc_alt_omega_L = cp.sum(cp.maximum(0.0, lin_stc_alt_omega_L))

    # ---------------------------------------------------------------

    _, lin_stc_alt_delta_U = f_alt_trig_delta_U(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_alt_delta_U = np.sum(np.maximum(0.0, lin_stc_alt_delta_U))
    else:
        lin_stc_alt_delta_U = cp.sum(cp.maximum(0.0, lin_stc_alt_delta_U))

    # ---------------------------------------------------------------

    _, lin_stc_alt_delta_L = f_alt_trig_delta_L(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_alt_delta_L = np.sum(np.maximum(0.0, lin_stc_alt_delta_L))
    else:
        lin_stc_alt_delta_L = cp.sum(cp.maximum(0.0, lin_stc_alt_delta_L))

    # ---------------------------------------------------------------

    _, lin_stc_alt_gs_U = f_alt_trig_gs_U(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_alt_gs_U = np.sum(np.maximum(0.0, lin_stc_alt_gs_U))
    else:
        lin_stc_alt_gs_U = cp.sum(cp.maximum(0.0, lin_stc_alt_gs_U))

    # ---------------------------------------------------------------

    _, lin_stc_alt_gs_L = f_alt_trig_gs_L(
        xk=X_last, uk=U_last, dx=X - X_last, du=U - U_last, params=params, npy=npy
    )

    if npy:
        lin_stc_alt_gs_L = np.sum(np.maximum(0.0, lin_stc_alt_gs_L))
    else:
        lin_stc_alt_gs_L = cp.sum(cp.maximum(0.0, lin_stc_alt_gs_L))

    # ---------------------------------------------------------------

    vehicle_cost = 0.0
    vehicle_cost += params["w_obj_fuel"] * (-X[0, -1])

    vehicle_cost += params["w_con_stl"] * (lin_stc_E1_L + lin_stc_E1_U)
    vehicle_cost += params["w_con_stl"] * (lin_stc_E2_L + lin_stc_E2_U)
    vehicle_cost += params["w_con_stl"] * (lin_stc_alt_spd + lin_stc_spd_spd)
    vehicle_cost += params["w_con_stl"] * (lin_stc_alt_theta_U + lin_stc_alt_theta_L)
    vehicle_cost += params["w_con_stl"] * (lin_stc_alt_omega_U + lin_stc_alt_omega_L)
    vehicle_cost += params["w_con_stl"] * (lin_stc_alt_delta_U + lin_stc_alt_delta_L)
    vehicle_cost += params["w_con_stl"] * (lin_stc_alt_gs_U + lin_stc_alt_gs_L)

    if npy and params["print_stl_cost"]:
        print("Fuel Cost      : ", X[0, -1])
        print("STC E1 L       : ", lin_stc_E1_L)
        print("STC E1 U       : ", lin_stc_E1_U)
        print("STC E2 L       : ", lin_stc_E2_L)
        print("STC E2 U       : ", lin_stc_E2_U)
        print("STC Alt Spd    : ", lin_stc_alt_spd)
        print("STC Spd Spd    : ", lin_stc_spd_spd)
        print("STC Alt Theta U: ", lin_stc_alt_theta_U)
        print("STC Alt Omega U: ", lin_stc_alt_omega_U)
        print("STC Alt Delta U: ", lin_stc_alt_delta_U)
        print("STC Alt Theta L: ", lin_stc_alt_theta_L)
        print("STC Alt Omega L: ", lin_stc_alt_omega_L)
        print("STC Alt Delta L: ", lin_stc_alt_delta_L)
        print("STC Alt Gs U   : ", lin_stc_alt_gs_U)
        print("STC Alt Gs L   : ", lin_stc_alt_gs_L)
        print("-" * 50)

    return vehicle_cost
