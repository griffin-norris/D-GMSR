import numpy as np
import typing as T


def qf_dynamics(params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:

    def f(x, u):
        speed = (x[3] ** 2 + x[4] ** 2 + x[5] ** 2) ** (0.5)
        return np.array(
            [
                [x[3]],
                [x[4]],
                [x[5]],
                [u[0] / params["m"] - params["cd"] * x[3] * speed],
                [u[1] / params["m"] - params["cd"] * x[4] * speed],
                [u[2] / params["m"] - params["cd"] * x[5] * speed - params["g0"]],
            ]
        )

    def A(x, u):
        speed = (x[3] ** 2 + x[4] ** 2 + x[5] ** 2 + 1e-6) ** (0.5)
        return np.array(
            [
                [0, 0, 0, 1.0, 0.0, 0.0],
                [0, 0, 0, 0.0, 1.0, 0.0],
                [0, 0, 0, 0.0, 0.0, 1.0],
                [
                    0,
                    0,
                    0,
                    -params["cd"] * (speed + x[3] * x[3] / speed),
                    -params["cd"] * (x[3] * x[4] / speed),
                    -params["cd"] * (x[3] * x[5] / speed),
                ],
                [
                    0,
                    0,
                    0,
                    -params["cd"] * (x[3] * x[4] / speed),
                    -params["cd"] * (speed + x[4] * x[4] / speed),
                    -params["cd"] * (x[4] * x[5] / speed),
                ],
                [
                    0,
                    0,
                    0,
                    -params["cd"] * (x[3] * x[5] / speed),
                    -params["cd"] * (x[4] * x[5] / speed),
                    -params["cd"] * (speed + x[5] * x[5] / speed),
                ],
            ]
        )

    def B(x, u):
        return np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1 / params["m"], 0, 0],
                [0, 1 / params["m"], 0],
                [0, 0, 1 / params["m"]],
            ]
        )

    params["f_func"] = f
    params["A_func"] = A
    params["B_func"] = B

    return params
