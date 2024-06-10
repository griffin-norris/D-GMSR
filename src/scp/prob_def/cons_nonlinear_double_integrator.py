import numpy as np
import cvxpy as cp
import typing as T


def ndi_cons_fcn(
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
    vehicle_cons += [cp.norm(X[2:4, 1:-1], axis=0) <= params["vehicle_v_max"]]
    vehicle_cons += [cp.norm(U[:, :], axis=0) <= params["vehicle_a_max"]]

    return vehicle_cons
