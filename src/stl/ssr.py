import sympy as sp
import typing as T


def ssr_sympy(k: int, K: int) -> T.Tuple[T.Any]:
    """
    :return: Smooth min and max and their derivatives
    """

    f = sp.zeros(2, 1)
    x = sp.Matrix(sp.symbols(["x" + str(i) for i in range(K)], real=True))

    def max_aprx(x, k, K):
        sum_1 = 0.0
        sum_2 = 0.0
        for i in range(K):
            sum_1 = sum_1 + x[i] * sp.exp(k * x[i])
            sum_2 = sum_2 + sp.exp(k * x[i])
        return sum_1 / sum_2

    def min_aprx(x, k, K):
        sum_1 = 0.0
        for i in range(K):
            sum_1 = sum_1 + sp.exp(k * -x[i])
        return -1 / k * sp.log(sum_1)

    f[0, 0] = min_aprx(x, k, K)
    f[1, 0] = max_aprx(x, k, K)

    A = f.jacobian(x)

    f_func = sp.lambdify((x,), f, "numpy")
    A_func = sp.lambdify((x,), A, "numpy")

    return f_func, A_func
