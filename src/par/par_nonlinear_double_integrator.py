import numpy as np
import typing as T


def ndi_params_fcn() -> T.Dict[str, T.Any]:
    """
    Static parameters for the scenario
    """

    t_f = 8.0  # Total simulation time [s]
    K = 17  # Total number of nodes
    t_scp = t_f / (K - 1)

    n_states = 4  # p_x, p_y, v_x, v_y
    n_controls = 2  # a_x, a_y

    x_init = np.array([0.0, 0.0, 0.0, 0.0])  # [x, y, x_dot, y_dot]
    x_final = np.array([8.0, 0.0, 0.0, 0.0])  # [x, y, x_dot, y_dot]

    X_last = np.zeros((n_states, K))
    U_last = np.zeros((n_controls, K))
    X_last[0, :] = np.linspace(x_init[0], x_final[0], K)

    return dict(
        # ---------------------------------------------------------------------------------------------------
        # General Parameters
        t_f=t_f,
        K=K,
        t_scp=t_scp,
        dt=t_scp,
        n_states=n_states,
        n_controls=n_controls,
        inp_param="FOH",
        x_init=x_init,
        x_final=x_final,
        X_last=X_last,
        U_last=U_last,
        vehicle_v_max=2.0,  # [m/s]
        vehicle_v_max_evnt=1.0,  # [m/s]
        vehicle_a_max=1.5,  # [m/s^2]
        cd=0.01,
        # ---------------------------------------------------------------------------------------------------
        # Prox-linear Parameters
        ite=20,  # Total number of iterations for the prox-linear algorithm
        w_obj_stl=1.0,  # Penalization weight for the actual cost
        w_con_dyn=1e3,  # Penalization weight for the constraints
        # Prox-linear via adaptive step-size
        adaptive_step=True,  # Updates the penalization cost of prox-linear
        w_ptr=1e-2,  # Initial penaliztion weight for the trust-regrion
        w_ptr_min=0.001,  # Minimum penaliztion weight for the trust-regrion
        r0=0.01,  # Minimum linearization accuracy to accept the SCP iteration and increase w_ptr
        r1=0.1,  # Minimum linearization accuracy to use the same w_ptr
        r2=0.9,  # Minimum linearization accuracy to decrease the w_ptr
        # ---------------------------------------------------------------------------------------------------
        # STL
        stl_eps=1e-8,
        stl_p=1,
        stl_w_phi_1=np.ones(K),
        stl_w_phi_2=np.ones(K),
        stl_w_phi_12=np.ones(2),
        p_w=np.array([[1.25], [-2.0], [0.2]]),  # Way-point
        # ---------------------------------------------------------------------------------------------------
        # Integration
        rk4_steps_dyn=5,
        rk4_steps_J=5,
        # ---------------------------------------------------------------------------------------------------
        print_ite_number=False,
    )
