from __future__ import annotations

import glob
import json
import logging
import os
import traceback

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from process_sliced_bubble import calculate_rmax
from process_sliced_bubble import clusterize_points
from process_sliced_bubble import get_time
from process_sliced_bubble import NumpyEncoder
from process_sliced_bubble import order_points_in_each_cluster
from process_sliced_bubble import plot_circle_with_curvature
from process_sliced_bubble import shift_to_xmin
from process_sliced_bubble import sort_names
from work.tube.pozrikidis import fit_curve, Config, full_shape_psi, full_shape_psi_x, fit_curve_err, sform

logging.Logger
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
)
logging.getLogger('matplotlib.font_manager').disabled = True

rc_params: dict = {
    # 'backend': 'pdf',
    'font.size': 8,
    'axes.labelsize': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'lines.linewidth': 0.5,
    # 'text.usetex': True,
    # 'text.latex.preamble': r'\usepackage{amsmath}',
    # "font.family": "serif",
    'text.usetex': False,  # Turn off LaTeX rendering
    "font.family": "Times New Roman",
    # 'ps.useafm': True,
    # 'pdf.use14corefonts': True,
    'figure.figsize': [32/5.33333 - 2*0.416667, 0.574],
    # 'figure.figsize': [32/5.33333/2 - 2*0.416667, 2],
}
grey = '#808080'
matplotlib.rcParams.update(rc_params)

# Function to recursively convert lists in a dictionary to NumPy arrays


def convert_lists_to_numpy_arrays(obj):
    if isinstance(obj, list):
        return np.asarray(obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = convert_lists_to_numpy_arrays(value)
    return obj


def find_nearest_index(array, value):
    if len(array) == 0:
        return 0
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if array[idx] < value:
        idx += 1
    return idx, array[idx] == value


if __name__ == "__main__":
    json_pattern = "metadata_t=*.json"
    path = os.path.join(os.getcwd(), "data_for_paper")
    xmin = -4.5
    xmax = 4.5
    ymin = -0.5
    ymax = 0.5
    width = 5.16666
    height = width/9

    # xmin = -1.51
    # xmax = 0.1
    # ymin = -0.9
    # ymax = 0.51
    # width = 32/5.33333/2 - 2*0.416667  # 2,1666
    # height = 2
    arrow_scale = 1
    vector_format = "pdf"
    props = {
        "mu1": 0.88e-3,
        "mu2": 0.019e-3,
        "rho1": 997,
        "rho2": 1.204,
        "sigma": 72.8e-3,
        "diam": 0.514e-3,
        "grav": 9.8,  # variable parameter
        "Umean": 4.117,
        "alpha": np.pi/6,  # variable parameter
        "s1": -1,
        "s2": 1,
        # "gradp_basilisk": 4.38820E5,  # from Poiseuille's flow
        "gradp_basilisk": 542_400,  # from p_drop
        # "d/diam": 1.295828280810274,  # variable parameter
        # "tail_y/diam": 0.2,
        # "Vd": 0.2179e-9,  # estimate volume
        # "a": 0.000373,  # estimate radius of drop
        # "l": 2.54107886e-04,  # estimate length of drop
    }
    logging.debug(f"props: {props}")

    json_names = glob.glob(os.path.join(path, "res27", json_pattern), recursive=False)

    json_names = sort_names(json_names)
    logging.debug(f"Found {json_pattern} files in: {json_names}")

    for ifile, file in enumerate(json_names):  # [::10]
        time = get_time(file)
        # Debugging
        # if time not in [7.03331]:  # 1.9472, 6.82811, #  5.43179, 7.19203, 10.3504
        #     continue
        # if time > 7.61:
        #     continue
        logging.debug(f"file: {file} time: {time}")
        try:
            with open(os.path.join(path, "res27", file)) as fd:
                res = json.load(fd)
                res = convert_lists_to_numpy_arrays(res)

            res["logger"] = logging
            shift_to_xmin(res)
            clusterize_points(res, width=0.02)
            order_points_in_each_cluster(res, max_retries=5, timeout_seconds=120)
            with open(os.path.join(path, "res27", file), "w") as f:
                json.dump(res, f, cls=NumpyEncoder)

            curvature0 = None
            circle_x = np.array([])
            circle_y = np.array([])
            coords_up = np.array([])
            coords_down = np.array([])
            plt.figure(figsize=(width, height))

            df = res["df"]
            tips = []
            for i, label in enumerate(res["labels"]):
                index_label = df["label"] == label
                xx, yy = df[index_label]["xp"].values, df[index_label]["yp"].values
                ux, umag = df[index_label]["ux"].values, df[index_label]["umag"].values
                n_points = len(xx)
                logging.info({"Processing label": label, "n_points": n_points})
                if n_points < 50:
                    res[label] = {
                        "xx": xx,
                        "yy": yy,
                        "new_xx": roots[0]["new_xx"],
                        "new_yy": roots[0]["new_yy"],
                        "roots": [],
                        "xmax": xx.max(),
                        "xmin": xx.min(),
                        "n_points": n_points,
                    }
                    continue
                roots = plot_circle_with_curvature(xx, yy, ux, umag, label, smooth_parameter=0.01)
                tips = tips + roots
                res[label] = {
                    "xx": xx,
                    "yy": yy,
                    "new_xx": roots[0]["new_xx"],
                    "new_yy": roots[0]["new_yy"],
                    "roots": roots,
                    "xmax": xx.max(),
                    "xmin": xx.min(),
                    "n_points": n_points,
                }
            tips = sorted(tips, key=lambda v: v.get("x0", -1000), reverse=True)
            second_tip = tips[1]
            # choose the second tip filter out too close tips
            for i in range(1, len(tips)):
                if tips[i].get("x0") is not None and abs(tips[i].get("x0") - tips[0].get("x0")) > 0.1:
                    second_tip = tips[i]
                    break

            # dataframe = pd.DataFrame({"x": second_tip["new_xx"], "y": second_tip["new_yy"]})
            # # take points of cluster "index" and above y>0
            # dataframe = dataframe[dataframe["y"] >= 0]
            # start = list(sorted(zip(dataframe["x"], dataframe["y"]), key=lambda x: x[0], reverse=True)[0])
            start = [second_tip["x0"], second_tip["y0"]]
            logging.debug(f"start:{start}")
            x_tip = start[0]
            start[0] -= x_tip
            res["xp"] -= x_tip
            for label in res["labels"]:
                res[label]["xx"] -= x_tip
                res[label]["new_xx"] -= x_tip
                for root in res[label]["roots"]:
                    if root.get("x0") is not None:
                        root["circle_x"] -= x_tip
                        root["x0"] -= x_tip
                        root["x_left"] -= x_tip
                        root["x_right"] -= x_tip
            # xmin = res["xp"].min()
            # xmax = res["xp"].max()
            rmax = calculate_rmax(second_tip, logging)
            second_tip["xx_left"] -= x_tip
            second_tip["x_left"] -= x_tip
            second_tip["xx_right"] -= x_tip
            second_tip["x_right"] -= x_tip
            second_tip["xx_peak_left"] -= x_tip
            second_tip["xx_peak_right"] -= x_tip
            second_tip["xx_left_point"] -= x_tip
            second_tip["xx_right_point"] -= x_tip
            curvature0 = np.abs(second_tip["curvature0"])
            logging.info({"second_tip": second_tip.keys()})
            # pendant drop
            props["second_tip"] = second_tip
            props["curvature"] = curvature0
            props["d"] = -1 or 0.5*(second_tip["xx_peak_left"] + second_tip["xx_peak_right"])
            if False:
                length_basilisk = np.sqrt(props["sigma"]/props["gradp_basilisk"])/props["diam"]
                config = Config(props, logging)
                _, Sigma_psi_basilisks = full_shape_psi(length_basilisk, config)
                X_psi_basilisk_full, Sigma_psi_basilisk_full = full_shape_psi_x(length_basilisk, config, mirror=True)

                length_guess = 0.48976214
                mode = "length"
                if mode == "length":
                    x_guess = (length_guess, )
                else:
                    x_guess = (length_guess, 2.0*curvature0)
                X_psi, Sigma_psi, _, _ = fit_curve(x_guess, config, mode=mode)
                X_psi_both, Sigma_psi_both = full_shape_psi(config.length_avg, config)
                X_psi_both_full, Sigma_psi_both_full = full_shape_psi_x(config.length_avg, config, mirror=True)

                logging.info({
                    "mode": mode,
                    "length_left": sform(config.length_left),
                    "length_right": sform(config.length_right),
                    "length_avg": sform(config.length_avg),
                    "length_basilisk": sform(length_basilisk),
                    "Rmax_left": sform(Sigma_psi[-1]),
                    "Rmax_right": sform(Sigma_psi[0]),
                    "Rmax_avg": sform(Sigma_psi_both[-1]),
                    "Rmax_basilisk": sform(Sigma_psi_basilisks[-1]),
                    "curvature": sform(config.curvature),
                    "B": sform(config.B),
                    "gradp_left": config.get_pressure_gradient(config.length_left),
                    "gradp_right": config.get_pressure_gradient(config.length_right),
                    "gradp_avg": config.get_pressure_gradient(config.length_avg),
                    "gradp_basilisk": config.get_pressure_gradient(length_basilisk),
                    "error_left": sform(fit_curve_err(x_guess, config, part="left", mode=mode)),
                    "error_right": sform(fit_curve_err(x_guess, config, part="right", mode=mode)),
                    "error_avg": sform(fit_curve_err(x_guess, config, part="both", mode=mode)),
                })

            # mode = "length+B"
            # if mode == "length":
            #     x_guess = (length_guess, )
            # else:
            #     x_guess = (length_guess, 2.0*curvature0)
            # X_psi_2, Sigma_psi_2, _, _ = fit_curve(x_guess, config, mode=mode)
            # X_psi_both_2, Sigma_psi_both_2 = full_shape_psi_x(config.length_avg, config, mirror=True)
            #
            # logging.info({
            #     "mode": mode,
            #     "length_left": sform(config.length_left),
            #     "length_right": sform(config.length_right),
            #     "length_avg": sform(config.length_avg),
            #     "length_basilisk": sform(length_basilisk),
            #     "curvature": sform(config.curvature),
            #     "B": sform(config.B),
            #     "gradp_left": config.get_pressure_gradient(config.length_left),
            #     "gradp_right": config.get_pressure_gradient(config.length_right),
            #     "gradp_avg": config.get_pressure_gradient(config.length_avg),
            #     "gradp_basilisk": config.get_pressure_gradient(length_basilisk),
            #     "error_left": sform(fit_curve_err(x_guess, config, part="left", mode=mode)),
            #     "error_right": sform(fit_curve_err(x_guess, config, part="right", mode=mode)),
            #     "error_avg": sform(fit_curve_err(x_guess, config, part="both", mode=mode)),
            # })



            ##########################
            ######### Draw ##########
            ##########################
            default_colors = [
                'blue', 'orange', 'green', 'brown', 'purple',
                'pink', 'grey', 'cyan', 'magenta', 'yellow', 'aqua',
            ]
            colors = dict()
            label_xmax = sorted(
                [(label, res[label]) for label in res["labels"]], key=lambda x: x[1].get(
                    "xmax",
                ) if x[1].get("n_points") > 100 else x[1].get("xmax") - 100, reverse=True,
            )
            for label, _ in label_xmax:
                colors[label] = default_colors.pop(0)

            for label in res["labels"]:
                # plt.scatter(res[label]["xx"], res[label]["yy"], s=1)
                # plt.plot(res[label]["xx"], res[label]["yy"], '-', color=colors[label])
                plt.plot(res[label]["new_xx"], res[label]["new_yy"], '-', color=colors[label], linewidth=0.8)
                # for root in res[label]["roots"]:
                #     plt.plot(root["circle_x"], root["circle_y"])
                #     plt.quiver(
                #         root["x0"], root["y0"], arrow_scale*root["nx0"], arrow_scale*root["ny0"],
                #         color='red', scale=100, width=0.0001
                #     )
                #     plt.scatter(root["x0"], root["y0"], s=0.1, color='red')
            # add fitting curve
            # plt.plot(X_psi, Sigma_psi, linestyle="dotted", color="lime", linewidth=0.8)
            # plt.plot(X_psi_2, Sigma_psi_2, linestyle="dotted", color="magenta", linewidth=0.8)
            # uncomment this:
            # plt.plot(X_psi_basilisk_full, Sigma_psi_basilisk_full, linestyle="solid", color="red", linewidth=0.5)
            # plt.plot(X_psi_x_basilisk , Sigma_psi_x_basilisk, linestyle="dotted", color="aqua", linewidth=0.8)
            # plt.plot(second_tip["xx_left"], second_tip["yy_left"], linestyle="-", color="pink")
            # plt.plot(second_tip["xx_right"], second_tip["yy_right"], linestyle="-", color="grey")
            # uncomment this:
            # plt.plot(X_psi_both_full, Sigma_psi_both_full, linestyle="solid", color="lime", linewidth=0.5)
            # plt.plot(X_psi_both, Sigma_psi_both, linestyle="solid", color="forestgreen", linewidth=0.5)

            # plt.plot(X_psi_both_2, Sigma_psi_both_2, linestyle="dotted", color="magenta", linewidth=0.8)

            plt.plot(
                second_tip["circle_x"], second_tip["circle_y"],
                linewidth=0.3, color="red", linestyle="dashed",
            )
            plt.quiver(
                second_tip["x0"], second_tip["y0"], arrow_scale *
                second_tip["nx0"], arrow_scale*second_tip["ny0"],
                color='red', scale=100, width=0.001,
            )
            plt.scatter(second_tip["x0"], second_tip["y0"], s=0.4, color='red')
            # Add tube horizontal lines
            plt.plot([xmin, xmax], [-0.5, -0.5], color='black', linewidth=1)
            plt.plot([xmin, xmax], [0.5, 0.5], color='black', linewidth=1)
            # Add text to the plot
            kappa_tail = r"\kappa_{\text{tail}}"
            rmax_drop = r"R_{\text{max,d}}"
            plt.title(
                label=rf'$t={time:.3f} \quad |{kappa_tail}|={curvature0:.3f} \quad {rmax_drop}={rmax:.3f}$',
                color='black',
                fontsize=6,
            )

            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            plt.grid(True, alpha=0.1, zorder=-1000)
            for direction in ['top', 'right', 'bottom', 'left']:
                plt.gca().spines[direction].set_color(grey)  # Grey color with opacity
            plt.tick_params('both', length=2, width=0.3, which='major', color=grey)
            # plt.tight_layout()
            plt.savefig(file[:-4] + "png", bbox_inches="tight", pad_inches=0, dpi=1200, transparent=False)
            plt.savefig(file[:-4] + vector_format, bbox_inches="tight", pad_inches=0, transparent=False)

            if len(second_tip.get("xx_peak_left", [])):
                xpeak = second_tip["xx_peak_left"]
                ypeak = second_tip["yy_peak_left"]
                plt.scatter(xpeak[0], ypeak[0], s=0.1, color='lime', zorder=10)
                plt.scatter(xpeak[1:], ypeak[1:], s=0.1, color='red', zorder=10)
                plt.scatter(
                    second_tip["xx_left_point"], second_tip["yy_left_point"], s=0.1, color='red', zorder=10
                )
            if len(second_tip.get("xx_peak_right", [])):
                xpeak = second_tip["xx_peak_right"]
                ypeak = second_tip["yy_peak_right"]
                plt.scatter(xpeak[-1], ypeak[-1], s=0.1, color='lime', zorder=10)
                plt.scatter(xpeak[:-1], ypeak[:-1], s=0.1, color='red', zorder=10)
                plt.scatter(
                    second_tip["xx_right_point"], second_tip["yy_right_point"], s=0.1, color='red', zorder=10
                )

            plt.scatter(second_tip["x_left"], second_tip["y_left"], s=0.1, color='black', zorder=10)
            plt.scatter(second_tip["x_right"], second_tip["y_right"], s=0.1, color='black', zorder=10)
            plt.savefig(
                "dots_" + file[:-4] + "png",
                bbox_inches="tight", pad_inches=0,  dpi=1200, transparent=False,
            )
            plt.savefig("dots_" + file[:-4] + vector_format, bbox_inches="tight", pad_inches=0, transparent=False)
            plt.close()

            # dump_data = {
            #     "simulation": {
            #         "x": res[0]["new_xx"],
            #         "y": res[0]["new_yy"]
            #     },
            #     "basilisk": {
            #         "x": X_psi_basilisk_full,
            #         "y": Sigma_psi_basilisk_full
            #     },
            #     "optimization": {
            #         "x": X_psi_both_full,
            #         "y": Sigma_psi_both_full
            #     }
            # }
            # with open(os.path.join(path, "figure_7.json"), "w") as f:
            #     json.dump(dump_data, f, cls=NumpyEncoder)

            # save after each calculation
            with open(os.path.join(path, "output_sliced_Rmax_U.json")) as fd:
                outputs = json.load(fd)
                i_time, replace_element = find_nearest_index(outputs["t"], time)
                if replace_element:
                    outputs["curvature_tip"][i_time] = second_tip["curvature0"]
                    outputs["U_tip"][i_time] = second_tip["umag"]
                    outputs["U_x"][i_time] = second_tip["ux"]
                    outputs["x_tip"][i_time] = second_tip["x0"]
                    outputs["nx"][i_time] = second_tip["nx0"]
                    outputs["ny"][i_time] = second_tip["ny0"]
                    outputs["rmax"][i_time] = rmax
                else:
                    outputs["t"].insert(i_time, time)
                    outputs["curvature_tip"].insert(i_time, second_tip["curvature0"])
                    outputs["U_tip"].insert(i_time, second_tip["umag"])
                    outputs["U_x"].insert(i_time, second_tip["ux"])
                    outputs["x_tip"].insert(i_time, second_tip["x0"])
                    outputs["nx"].insert(i_time, second_tip["nx0"])
                    outputs["ny"].insert(i_time, second_tip["ny0"])
                    outputs["rmax"].insert(i_time, rmax)
            with open(os.path.join(path, "output_sliced_Rmax_U.json"), "w") as fd:
                json.dump(outputs, fd, cls=NumpyEncoder)

        except Exception as e:
            logging.error({
                "error": "Failed to process json",
                "message": e,
                "t": time,
                "traceback": traceback.format_exc(),
            })
            continue
