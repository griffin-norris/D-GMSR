import numpy as np
import sympy as sp
import typing as T


def gmsr_sympy(eps: float, p: int, weights: np.ndarray, N: int) -> T.Tuple[T.Any]:
    """
    :return: Functions to calculate f and A given state x
    """

    f = sp.zeros(2, 1)
    x = sp.Matrix(sp.symbols(["x" + str(i) for i in range(N)], real=True))

    fp_2 = sp.Matrix([sp.Max(0, i) ** 2 for i in x])
    fn_2 = sp.Matrix([sp.Min(0, i) ** 2 for i in x])

    fp_2p = sp.Matrix([sp.Max(0, i) ** (2 * p) for i in x])
    fn_2p = sp.Matrix([sp.Min(0, i) ** (2 * p) for i in x])

    M0_fp = 1
    M0_fn = 1

    Mp_fp = 0
    Mp_fn = 0

    sum_w = np.sum(weights)

    for i in range(N):
        w_i = int(weights[i])

        M0_fp = M0_fp * (fp_2[i] ** w_i)
        M0_fn = M0_fn * (fn_2[i] ** w_i)

        Mp_fp = Mp_fp + w_i * fp_2p[i] / sum_w
        Mp_fn = Mp_fn + w_i * fn_2p[i] / sum_w

    M0_fp = (eps ** (sum_w) + M0_fp) ** (1 / sum_w)
    M0_fn = (eps ** (sum_w) + M0_fn) ** (1 / sum_w)

    Mp_fp = (eps ** (p) + Mp_fp) ** (1 / p)
    Mp_fn = (eps ** (p) + Mp_fn) ** (1 / p)

    f[0, 0] = M0_fp ** (1 / 2) - Mp_fn ** (1 / 2)  # {}^{\wedge} h_{p, w}^{\epsilon}
    f[1, 0] = Mp_fp ** (1 / 2) - M0_fn ** (1 / 2)  # {}^{\vee}   h_{p, w}^{\epsilon}

    A = f.jacobian(x)

    f_func = sp.lambdify((x,), f, "numpy")
    A_func = sp.lambdify((x,), A, "numpy")

    return f_func, A_func


def gmsr_and(
    eps: float,
    p: int,
    weights: np.ndarray,
    *args: T.Tuple[T.Any],
) -> T.Tuple[T.Any]:
    """
    Computes the generalized mean-based smooth robustness (GMSR) measure for a logical AND operation
    and its gradient based on input function values and their gradients.

    Parameters:
    eps (float): A small positive constant used for numerical stability.
    p (int): An integer parameter that influences the calculation of the GMSR.
    weights (np.ndarray): An array of weights for the input function values.
    args (T.Tuple[T.Any]): A tuple containing the values of the functions and their gradients
                           to be combined using the AND operation.

    Returns:
    T.Tuple[T.Any]: A tuple containing the GMSR value of the AND operation and its gradient.
    """

    # Number of functions
    K = len(args[0])
    fcn_vals = args[0]

    # Split indices into positive and non-positive values
    pos_idx = list(idx for idx, ele in enumerate(fcn_vals) if ele > 0.0)
    neg_idx = list(idx for idx, ele in enumerate(fcn_vals) if ele <= 0.0)

    # Extract values and weights for positive and non-positive indices
    pos_vals = fcn_vals[pos_idx]
    neg_vals = fcn_vals[neg_idx]
    pos_w = weights[pos_idx]
    neg_w = weights[neg_idx]

    # Sum of all weights
    sum_w = np.array(weights).sum()

    if neg_idx:
        # If there exists a negative element
        # Calculate the GMSR value (h_and) and its gradient (c_i_w_i)

        # Function value calculation
        sums = 0.0
        for idx, neg_val in enumerate(neg_vals):
            sums += neg_w[idx] * (neg_val ** (2 * p))
        Mp = (eps**p + (sums / sum_w)) ** (1 / p)
        h_and = eps**0.5 - Mp**0.5

        # Gradient calculation
        cp = 0.5 * Mp**-0.5
        cpm = 2 * p / (p * sum_w * Mp ** (p - 1))
        c_i_w_i = np.zeros(K)
        c_i_w_i[neg_idx] = [
            cp * cpm * (neg_w[idx] * (np.abs(neg_val)) ** (2 * p - 1))
            for idx, neg_val in enumerate(neg_vals)
        ]

    else:
        # If all values are positive
        # Calculate the GMSR value (h_and) and its gradient (c_i_w_i)

        # Function value calculation
        mult = 1.0
        for idx, pos_val in enumerate(pos_vals):
            mult *= pos_val ** (2 * pos_w[idx])
        M0 = (eps**sum_w + mult) ** (1 / sum_w)
        h_and = M0**0.5 - eps**0.5

        # Gradient calculation
        c0 = 0.5 * M0**-0.5
        c0m = (2 * mult) / (sum_w * M0 ** (sum_w - 1))
        c_i_w_i = np.zeros(K)
        c_i_w_i[pos_idx] = [
            c0 * c0m * (pos_w[idx] / pos_val) for idx, pos_val in enumerate(pos_vals)
        ]

    return h_and, c_i_w_i


def gmsr_or(
    eps: float,
    p: int,
    weights: np.ndarray,
    *args: T.Tuple[T.Any],
) -> T.Tuple[T.Any]:
    """
    Computes the generalized mean-based smooth robustness (GMSR) measure for a logical OR operation
    and its gradient based on input function values and their gradients.

    Parameters:
    eps (float): A small positive constant used for numerical stability.
    p (int): An integer parameter that influences the calculation of the GMSR.
    weights (np.ndarray): An array of weights for the input function values.
    args (T.Tuple[T.Any]): A tuple containing the values of the functions and their gradients
                           to be combined using the OR operation.

    Returns:
    T.Tuple[T.Any]: A tuple containing the GMSR value of the OR operation and its gradient.
    """

    # Negate the function values
    args = -args[0]

    # Use the gmsr_and function to compute the GMSR value and gradient for the negated values
    h_mor, d_i_w_i = gmsr_and(eps, p, weights, args)

    # Return the negated GMSR value and the original gradient
    return -h_mor, d_i_w_i


def verify_gmsr_fcn(ITE: int):

    err_arr = np.zeros(8)
    for _ in range(ITE):

        K = np.random.randint(10, 20)
        p = np.random.randint(1, 8)
        eps = 0.1 * np.random.rand()
        weights = np.random.randint(1, 10, size=K)

        f_func, A_func = gmsr_sympy(eps, p, weights, K)

        y = np.random.rand(K)
        h_and, h_and_grad = gmsr_and(eps, p, weights, y)
        h_or, h_or_grad = gmsr_or(eps, p, weights, y)

        err_arr[0] = (f_func(y)[0, :] - h_and).sum()
        err_arr[1] = (A_func(y)[0, :] - h_and_grad).sum()
        err_arr[2] = (f_func(y)[1, :] - h_or).sum()
        err_arr[3] = (A_func(y)[1, :] - h_or_grad).sum()

        x = y - 0.5
        h_and, h_and_grad = gmsr_and(eps, p, weights, x)
        h_or, h_or_grad = gmsr_or(eps, p, weights, x)

        err_arr[4] = (f_func(x)[0, :] - h_and).sum()
        err_arr[5] = (A_func(x)[0, :] - h_and_grad).sum()
        err_arr[6] = (f_func(x)[1, :] - h_or).sum()
        err_arr[7] = (A_func(x)[1, :] - h_or_grad).sum()

        if err_arr.sum() > 1e-8:
            print(err_arr.sum())
            print("Implementation of D-GMSR is wrong!")
            break

    print("D-GMSR - Successful")
