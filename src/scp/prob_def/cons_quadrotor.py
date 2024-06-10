import numpy as np
import cvxpy as cp
import typing as T

def qf_cons_fcn(
    X: np.ndarray,
    U: np.ndarray,
    X_last: np.ndarray,
    U_last: np.ndarray,
    params: T.Dict[str, T.Any],
    npy: bool,
) -> T.Tuple[T.Any]:

    vehicle_cons = []
    vehicle_cons += [X[:, 0] == params["x_init"]]
    vehicle_cons += [X[:, -1] == params["x_final"]]
    vehicle_cons += [cp.norm(X[3:6, 1:-1], axis=0) <= params["vehicle_v_max"]]
    vehicle_cons += [cp.norm(U[:, 1:-1], axis=0) <= params["vehicle_T_max"]]
    vehicle_cons += [U[:, 0] == np.array((0.0, 0.0, params["m"] * params["g0"]))]
    vehicle_cons += [U[:, -1] == np.array((0.0, 0.0, params["m"] * params["g0"]))]
    vehicle_cons += [
        cp.norm(U[0:3, 1:-1], axis=0)
        <= U[2, 1:-1] / np.cos(params["vehicle_theta_max"])
    ]

    return vehicle_cons
