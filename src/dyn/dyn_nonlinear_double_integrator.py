import numpy as np
import typing as T


def ndi_dynamics(params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:

    def f(x, u):
        speed = (x[2] ** 2 + x[3] ** 2) ** (0.5)
        return np.array(
            [
                [x[2]],
                [x[3]],
                [u[0] - params["cd"] * x[2] * speed],
                [u[1] - params["cd"] * x[3] * speed],
            ]
        )

    def A(x, u):
        speed = (x[2] ** 2 + x[3] ** 2 + 1e-6) ** (0.5)
        return np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [
                    0,
                    0,
                    -params["cd"] * (speed + x[2] * x[2] / speed),
                    -params["cd"] * (x[2] * x[3] / speed),
                ],
                [
                    0,
                    0,
                    -params["cd"] * (x[2] * x[3] / speed),
                    -params["cd"] * (speed + x[3] * x[3] / speed),
                ],
            ]
        )

    def B(x, u):
        return np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

    params["f_func"] = f
    params["A_func"] = A
    params["B_func"] = B

    return params
