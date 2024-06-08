import numpy as np
import cvxpy as cp
import typing as T

from src.dis.integration import integrate_multiple
from src.dis.discretization import calculate_discretization

from .scp_nonlinear_cost import scp_non_linear_cost


def solve_convex_problem(
    X_last: np.ndarray,
    U_last: np.ndarray,
    w_tr: float,
    params: T.Dict[str, T.Any],
) -> T.Tuple[np.ndarray]:
    """
    Solves the convex sub-problem using ECOS or MOSEK and retruns the optimal values of X and U
    """

    X = cp.Variable((params["n_states"], params["K"]))
    U = cp.Variable((params["n_controls"], params["K"]))
    nu = cp.Variable((params["n_states"], params["K"] - 1))

    cost = w_tr * (
        cp.sum(cp.norm(X - X_last, axis=0) ** 2)
        + cp.sum(cp.norm(U - U_last, axis=0) ** 2)
    )  # Trust region
    cost += params["w_con_dyn"] * cp.sum(cp.abs(nu))

    constraints = [
        X[:, k + 1]
        == cp.reshape(params["A_bar"][:, k], (params["n_states"], params["n_states"]))
        @ X[:, k]
        + cp.reshape(params["B_bar"][:, k], (params["n_states"], params["n_controls"]))
        @ U[:, k]
        + cp.reshape(params["C_bar"][:, k], (params["n_states"], params["n_controls"]))
        @ U[:, k + 1]
        + params["z_bar"][:, k]
        + nu[:, k]
        for k in range(params["K"] - 1)
    ]

    cost += params["vehicle_cost_fcn"](X, U, X_last, U_last, params, npy=False)
    constraints += params["vehicle_cons_fcn"](X, U, X_last, U_last, params, npy=False)

    # Create the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)

    # Solve the optimization problem
    try:
        problem.solve(solver="ECOS")
    except:
        problem.solve(solver="MOSEK")

    return X.value, U.value, nu.value


def prox_linear(
    params: T.Dict[str, T.Any],
) -> T.Dict[str, T.Any]:
    """
    Solves the non-convex trajectory optimization problem using Penalized Trust Region method (PTR)
    """

    X_last = params["X_last"]
    U_last = params["U_last"]
    w_tr = params["w_ptr"]
    last_cost = None

    prox_results = dict()
    prox_results["ptr_cost_list"] = []
    prox_results["non_lin_cost_list"] = []
    prox_results["dynamical_cost_list"] = []
    prox_results["vehicle_cost_list"] = []
    prox_results["w_tr_list"] = []

    for i in range(params["ite"]):
        if params["print_ite_number"]:
            print("-" * 50)
            print("ite: ", i)
            print("-" * 50)
            print(" ")
        params = calculate_discretization(X_last, U_last, params)
        while True:
            if w_tr > 1e8:
                break

            X_new, U_new, nu_new = solve_convex_problem(X_last, U_last, w_tr, params)

            X_nl = X_new.copy()
            X_nl[:, 1:] = integrate_multiple(X_new, U_new, params)

            cost_dict = scp_non_linear_cost(
                X_new, U_new, X_last, U_last, nu_new, (X_new - X_nl), w_tr, params
            )

            if not (last_cost) or not (params["adaptive_step"]):
                X_last = X_new.copy()
                U_last = U_new.copy()
                last_cost = cost_dict["n_lin_cost"]

                prox_results["ptr_cost_list"].append(cost_dict["ptr_cost"])
                prox_results["non_lin_cost_list"].append(cost_dict["n_lin_cost"])
                prox_results["dynamical_cost_list"].append(
                    np.sum(np.linalg.norm((X_new - X_nl)[:, :], axis=1, ord=1))
                )
                prox_results["vehicle_cost_list"].append(
                    cost_dict["n_lin_vehicle_cost"]
                )
                prox_results["w_tr_list"].append(w_tr)
                break

            else:
                delta_J = last_cost - cost_dict["n_lin_cost"]
                delta_L = last_cost - cost_dict["lin_cost"]

                rho = delta_J / delta_L
                if rho <= params["r0"]:
                    w_tr = w_tr * 2

                else:
                    X_last = X_new.copy()
                    U_last = U_new.copy()
                    last_cost = cost_dict["n_lin_cost"]

                    prox_results["ptr_cost_list"].append(cost_dict["ptr_cost"])
                    prox_results["non_lin_cost_list"].append(cost_dict["n_lin_cost"])
                    prox_results["dynamical_cost_list"].append(
                        np.sum(np.linalg.norm((X_new - X_nl)[:, :], axis=1, ord=1))
                    )
                    prox_results["vehicle_cost_list"].append(
                        cost_dict["n_lin_vehicle_cost"]
                    )
                    prox_results["w_tr_list"].append(w_tr)

                    if params["r2"] <= rho:
                        w_tr = np.maximum(w_tr / 2, params["w_ptr_min"])
                    break

    prox_results["X_new"] = X_last
    prox_results["U_new"] = U_last

    return prox_results
