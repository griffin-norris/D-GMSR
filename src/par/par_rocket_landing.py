import numpy as np
import typing as T


def rl_params_fcn() -> T.Dict[str, T.Any]:
    """
    Static parameters for the scenario
    """

    t_f = 20  # Total simulation time [s]
    K = 21  # Total number of nodes

    n_states = 8
    n_controls = 4

    # 'm rx rz vx vz theta omega delta'
    x_init = np.array(
        [100000.0, 500.0, 100.0, -85.0, 0.0, 90.0 * np.pi / 180, 0.0, 0.0]
    )
    x_final = np.array([85000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    T_max = 1.00 * 2200 * 1e3  # Max thrust [N]
    T_min = 0.40 * 2200 * 1e3  # Min thrust [N]

    # Initialization
    X_last = np.zeros((n_states, K))
    U_last = np.zeros((n_controls, K))
    X_last = np.linspace(x_init, x_final, K).T

    U_last[0, :] = np.ones(K) * (T_max + T_min) / 2
    U_last[1, :] = np.ones(K) * (T_max + T_min) / 2
    U_last[2, :] = np.ones(K) * (T_max + T_min) / 2

    return dict(
        # ---------------------------------------------------------------------------------------------------
        # General Parameters
        t_f=t_f,
        K=K,
        t_scp=t_f / (K - 1),
        dt=0.1,
        n_states=n_states,
        n_controls=n_controls,
        inp_param="FOH",
        x_init=x_init,
        x_final=x_final,
        X_last=X_last,
        U_last=U_last,
        g_0=9.806,  # Earth's gravity at sea-level [m/s^2]
        I_sp=330,  # Rocket engine specific impulse [s]
        l_r=4.5,  # Radius of the fusulage [m]
        l_h=50,  # Height of the fusulage [m]
        l_cm=0.4 * 50,  # The distance between the cg and the engine [m]
        l_cp=0.2 * 50,  # The distance between the cp and the engine [m]
        rho_air=1.225,  # Air density [kg m^{-3}]
        S_area=545,  # Area of the rocket [m^{2}]
        C_aero=np.array([[0.0522, 0.0], [0.0, 0.4068]]),  # Drag coefficient
        m_f=85000.0,  # Final mass of the rocket [kg]
        min_alt=0.0,  # Min altitude [m]
        T_max=T_max,  # Max thrust [N]
        T_min=T_min,  # Min thrust [N]
        delta_max=10 * np.pi / 180,  # Max gimbal angle [deg]
        delta_dot_max=15 * np.pi / 180,  # Max gimbal angle rate [deg/s]
        spd_trig=35,  # Tigeer speed for the Engine - 2 and - 3
        alt_trig=100,  # Tigeer altitude for the speed - angle - angle rate - gimbal - glideslope
        spd_trig_aft=20.0,  # Final max velocity
        theta_trig_aft=5.0 * np.pi / 180,  # Max angle [deg]
        omega_trig_aft=2.5 * np.pi / 180,  # Max angle rate [deg/s]
        delta_trig_aft=1 * np.pi / 180,  # Max gimbal angle rate [deg/s]
        gs_trig_aft=5 * np.pi / 180,  # Glide slope angle [deg]
        # ---------------------------------------------------------------------------------------------------
        # Prox-linear Parameters
        ite=17,  # Total number of iterations for the prox-linear algorithm
        w_con_dyn=1e3,  # Penalization weight for the constraints
        w_obj_fuel=100,  # Penalization weight for the fuel consumption cost
        w_con_stl=1,  # Penalization weight for the stl constraints
        # Prox-linear via adaptive step-size
        adaptive_step=True,  # Updates the penalization cost of prox-linear
        w_ptr=8,  # Initial penaliztion weight for the trust-regrion
        w_ptr_min=0.001,  # Minimum penaliztion weight for the trust-regrion
        r0=0.01,  # Minimum linearization accuracy to accept the SCP iteration and increase w_ptr
        r1=0.1,  # Minimum linearization accuracy to use the same w_ptr
        r2=0.9,  # Minimum linearization accuracy to decrease the w_ptr
        # ---------------------------------------------------------------------------------------------------
        # STL
        stl_eps=1e-8,
        stl_p=1,
        stl_w_phi=np.ones(2),
        # ---------------------------------------------------------------------------------------------------
        # Integration
        rk4_steps_dyn=10,
        rk4_steps_J=10,
        # Scale params
        rva_sc=1e2,
        tm_sc=1e5,
        # ---------------------------------------------------------------------------------------------------
        print_ite_number=False,
        print_stl_cost=False,
    )


def scale_params(params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:

    rva_sc = params["rva_sc"]
    tm_sc = params["tm_sc"]

    params["x_init"][1:4] = params["x_init"][1:4] / (rva_sc)
    params["x_final"][1:4] = params["x_final"][1:4] / (rva_sc)

    params["g_0"] = params["g_0"] / (rva_sc)
    params["l_r"] = params["l_r"] / (rva_sc)
    params["l_h"] = params["l_h"] / (rva_sc)
    params["l_cm"] = params["l_cm"] / (rva_sc)
    params["l_cp"] = params["l_cp"] / (rva_sc)
    params["min_alt"] = params["min_alt"] / (rva_sc)
    params["spd_trig"] = params["spd_trig"] / (rva_sc)
    params["alt_trig"] = params["alt_trig"] / (rva_sc)
    params["spd_trig_aft"] = params["spd_trig_aft"] / (rva_sc)

    params["x_init"][0] = params["x_init"][0] / tm_sc
    params["x_final"][0] = params["x_final"][0] / tm_sc
    params["m_f"] = params["m_f"] / tm_sc

    params["U_last"][0:3, :] = params["U_last"][0:3, :] / (tm_sc * rva_sc)
    params["T_max"] = params["T_max"] / (tm_sc * rva_sc)
    params["T_min"] = params["T_min"] / (tm_sc * rva_sc)

    params["rho_air"] = params["rho_air"] / (rva_sc * tm_sc) * (rva_sc * rva_sc)

    params["X_last"] = np.linspace(params["x_init"], params["x_final"], params["K"]).T

    return params


def unscale_results(
    results: T.Dict[str, T.Any], params: T.Dict[str, T.Any]
) -> T.Dict[str, T.Any]:

    rva_sc = params["rva_sc"]
    tm_sc = params["tm_sc"]

    results["x_all"][:, 0] = tm_sc * results["x_all"][:, 0]
    results["x_all"][:, 1:5] = rva_sc * results["x_all"][:, 1:5]
    results["u_all"][:, 0:3, :] = tm_sc * rva_sc * results["u_all"][:, 0:3, :]

    results["x_nmpc_all"][0, :, 0] = tm_sc * results["x_nmpc_all"][0, :, 0]
    results["x_nmpc_all"][0, :, 1:5] = rva_sc * results["x_nmpc_all"][0, :, 1:5]
    results["u_nmpc_all"][0, :, 0:3] = tm_sc * rva_sc * results["u_nmpc_all"][0, :, 0:3]

    return results
