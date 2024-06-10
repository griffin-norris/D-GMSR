import matplotlib.pyplot as plt


def plot_cost(results, params, vehicle):

    fig, (ax0, ax1, ax2) = plt.subplots(figsize=(6, 6), nrows=3, sharex=True)

    ax0.plot(results["dynamical_cost_list_all"][0, :])
    ax0.set_xlabel(r"iterations", fontsize=12)
    ax0.set_ylabel("Dynamical Cost", fontsize=12)
    ax0.set_yscale("log")

    fuel_cost = 0.0
    if vehicle == "rl":
        fuel_cost = (
            params["w_obj_fuel"] * results["x_nmpc_all"][0, -1, 0] / params["tm_sc"]
        )

    ax1.plot(results["vehicle_cost_list_all"][0, :] + fuel_cost)

    ax1.set_xlabel(r"iterations", fontsize=12)
    ax1.set_ylabel(r"STL Cost", fontsize=12)

    ax2.plot(results["w_tr_list_all"][0, :])
    ax2.set_xlabel(r"iterations", fontsize=12)
    ax2.set_ylabel(r"TR Weight", fontsize=12)

    print("Dynamical Cost", results["dynamical_cost_list_all"][0, -1])
    print("STL Cost", results["vehicle_cost_list_all"][0, -1] + fuel_cost)
    print("TR Weight", results["w_tr_list_all"][0, -1])
