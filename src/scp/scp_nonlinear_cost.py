import numpy as np
import typing as T


def scp_non_linear_cost(
    X_new: np.ndarray,
    U_new: np.ndarray,
    X_last: np.ndarray,
    U_last: np.ndarray,
    nu_new: np.ndarray,
    nl_nu_new: np.ndarray,
    w_tr: float,
    params: T.Dict[str, T.Any],
) -> T.Tuple[np.ndarray]:
    """
    Returns the nonlinear or linearized cost value of the SCP problem
    Required for prox-linear with adaptive step-size
    """

    cost_dict = dict()

    cost_dict["ptr_cost"] = (
        w_tr
        * (
            np.linalg.norm(X_new - X_last, axis=0) ** 2
            + np.linalg.norm(U_new - U_last, axis=0) ** 2
        ).sum()
    )

    cost_dict["lin_cost"] = cost_dict["ptr_cost"]
    cost_dict["n_lin_cost"] = cost_dict["ptr_cost"]

    cost_dict["lin_vehicle_cost"] = params["vehicle_cost_fcn"](
        X_new, U_new, X_last, U_last, params, npy=True
    )
    cost_dict["n_lin_vehicle_cost"] = params["vehicle_cost_fcn"](
        X_new, U_new, X_new, U_new, params, npy=True
    )

    cost_dict["lin_cost"] += cost_dict["lin_vehicle_cost"]
    cost_dict["n_lin_cost"] += cost_dict["n_lin_vehicle_cost"]

    cost_dict["lin_dyn_cost"] = params["w_con_dyn"] * np.linalg.norm(
        nu_new.reshape(-1), 1
    )
    cost_dict["n_lin_dyn_cost"] = params["w_con_dyn"] * np.linalg.norm(
        nl_nu_new.reshape(-1), 1
    )

    cost_dict["lin_cost"] += cost_dict["lin_dyn_cost"]
    cost_dict["n_lin_cost"] += cost_dict["n_lin_dyn_cost"]

    return cost_dict
