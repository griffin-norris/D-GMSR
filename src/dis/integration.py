import numpy as np
import typing as T


def rk4(
    func: T.Any, y0: np.ndarray, tf: float, steps: int, *args: T.Tuple[T.Any]
) -> np.ndarray:
    """
    Implementation of the fourth-order Runge-Kutta (RK4) method for numerical integration.

    Parameters:
    - f: Function representing the system of ordinary differential equations (ODEs).
    - y0: Initial conditions (numpy array, n-dimensional column vector).
    - t: Time points for which the solution is calculated.

    Returns:
    - y: Solution of the ODEs at each time point.
    """

    t = np.linspace(0, tf, int(steps))  # Time points

    # Ensure y0 is a NumPy array (n-dimensional column vector)
    # y0 = np.array(y0).reshape(-1, 1)
    y0 = y0.reshape(-1, 1)

    # Initialize solution array
    y = np.zeros((len(t), len(y0)))
    y[0, :] = y0.flatten()

    # Perform RK4 integration
    for i in range(len(t) - 1):

        h = t[i + 1] - t[i]
        k1 = h * func(y[i], t[i], args)
        k2 = h * func(y[i] + 0.5 * k1, t[i] + 0.5 * h, args)
        k3 = h * func(y[i] + 0.5 * k2, t[i] + 0.5 * h, args)
        k4 = h * func(y[i] + k3, t[i] + h, args)

        y[i + 1, :] = y[i, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y


def dxdt(
    x: np.ndarray,
    t: float,
    *args: T.Tuple[T.Any],
) -> np.ndarray:
    """
    return: x_dot(t)
    """

    tt, tf, u_0, u_1, params = args[0]

    if params["inp_param"] == "FOH":
        u = u_0 + (t / tf) * (u_1 - u_0)
    elif params["inp_param"] == "ZOH":
        u = u_0.copy()

    return params["f_func"](x, u)[:, 0]


def integrate_dynamics(
    x: np.ndarray,
    u_0: np.ndarray,
    u_1: np.ndarray,
    params: T.Dict[str, T.Any],
    tf: float,
    tt: float,
) -> T.Tuple[np.ndarray]:
    """
    Integration of the vehicle dynamics [0, tf]
    return: x[t+dt] and u
    """
    x_next = rk4(dxdt, x, tf, params["rk4_steps_dyn"], tt, tf, u_0, u_1, params)[-1, :]
    return x_next


def integrate_multiple(X, U, params):

    x_k1_list = []
    for k in range(X.shape[1] - 1):
        tt = k * params["t_scp"]
        x_k1 = integrate_dynamics(
            X[:, k], U[:, k], U[:, k + 1], params, params["t_scp"], tt
        )
        x_k1_list.append(x_k1)

    return (np.vstack(x_k1_list)).T
