from __future__ import annotations

import glob
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from process_sliced_bubble import compute_curvature
from process_sliced_bubble import compute_curvature_from_ordered_points
from process_sliced_bubble import compute_normals_from_ordered_points
from process_sliced_bubble import find_df_centers
from process_sliced_bubble import get_time
from process_sliced_bubble import optimized_path
from process_sliced_bubble import order_points_in_each_cluster
from process_sliced_bubble import plot_circle
from process_sliced_bubble import plot_circle_with_curvature
from process_sliced_bubble import sort_names
from scipy import interpolate
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Function to recursively convert lists in a dictionary to NumPy arrays
def convert_lists_to_numpy_arrays(obj):
    if isinstance(obj, list):
        return np.asarray(obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = convert_lists_to_numpy_arrays(value)
    return obj


# find point where it start to decrease
def find_extremum(coords, find_decrease=True, eps=1e-3):
    # my_list_x = coords[:, 0]
    my_list_y = coords[:, 1]
    my_result = len(my_list_y) - 1
    if find_decrease:
        peaks, _ = find_peaks(my_list_y, height=0, distance=100)
        logging.debug(
            f"find_decrease peaks={peaks}  my_list_y={my_list_y[peaks]}",
        )
        my_result = peaks[0]
        # for index in range(1, len(my_list_y) - 1):
        #     # if my_list_x[index] < -0.01:
        #     #     continue
        #     if my_list_y[index] - my_list_y[index + 1] > eps:
        #         my_result = index
        #         break
    else:
        peaks, _ = find_peaks(-my_list_y, height=0, distance=100)
        logging.debug(
            f"find_increasing peaks={peaks}  my_list_y={my_list_y[peaks]}",
        )
        my_result = peaks[0]
        # for index in range(1, len(my_list_y) - 1):
        #     # if my_list_x[index] < -0.01:
        #     #     continue
        #     if my_list_y[index + 1] - my_list_y[index] > eps:
        #         my_result = index
        #         break
    return my_result


def give_coord(x, y, df_in, index, side="up"):
    df = df_in.copy(deep=True)
    # find the start point
    # df_cluster = df[(df['label'] == index)] # take points of cluster "index"
    if side == "up":
        # take points of cluster "index" and above y>0
        df_ind = df[(df["label"] == index) & (df["y"] > 0)]
    else:
        # take points of cluster "index" and above y>0
        df_ind = df[(df["label"] == index) & (df["y"] <= 0)]
    # sort by x and take the last right element
    start = list(
        sorted(
            zip(df_ind["x"], df_ind["y"]),
            key=lambda x: x[0],
        )[-1],
    )
    logging.debug(f"start:{start}")

    # x_tip = start[0]
    # x -= x_tip
    # df['x'] -= x_tip
    # df_cluster['x'] -= x_tip
    # start[0] -= x_tip

    coords = np.c_[x, y]
    if side == "up":
        coords = coords[coords[:, 1] > 0]
    else:
        coords = coords[coords[:, 1] <= 0]
    clustering = DBSCAN(eps=0.1, min_samples=2).fit(coords)
    logging.debug(f"Found N={len(set(clustering.labels_))} clusters")
    df_compare = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], "label": clustering.labels_},
    )
    # logging.debug(f'df: {df_compare}')
    label = df_compare[(df_compare["x"] == start[0]) & (
        df_compare["y"] == start[1]
    )]["label"].iloc[0]
    logging.debug(f"label: {label}")
    ind = df_compare["label"] == label
    coords = np.c_[df_compare["x"][ind], df_compare["y"][ind]]
    # coords = optimized_path(coords, start)
    return coords


json_pattern = "metadata_t=*.json"
path = os.getcwd()
xmin = -4
xmax = 4
ymax = 1.5
picScale = 4
picScale1 = 16

props = dict()
props["mu1"] = 0.88e-3
props["mu2"] = 0.019e-3
props["rho1"] = 997
props["rho2"] = 1.204
props["sigma"] = 72.8e-3
props["diam"] = 0.514e-3
props["grav"] = 9.8  # variable parameter
props["alpha"] = 110 * np.pi / 180  # variable parameter
props["d/diam"] = 1.295828280810274  # variable parameter
props["s1"] = -1
props["s2"] = 1
props["Vd"] = 0.2179e-9  # estimate volume
props["a"] = 0.000373  # estimate radius of drop
logging.debug(f"props={props}")

json_names = glob.glob(os.path.join(path, json_pattern), recursive=False)

json_names = sort_names(json_names)
logging.debug(f"Found {json_pattern} files in: {json_names}")

outputs: dict = {
    "t": [],
    "curvature_tip": [],
    "U_tip": [],
    "x_tip": [],
    "rmax": [],
}

for ifile, file in enumerate(json_names):
    time = get_time(file)
    # Debugging
    if time != 5.91581:
        continue
    # if time > 7.61:
    #     continue
    logging.debug(f"file: {file} time: {time}")
    try:
        with open(file) as fd:
            res = json.load(fd)
            res = convert_lists_to_numpy_arrays(res)
    except Exception:
        continue
    left_x = res["xp"].min()
    right_x = res["xp"].max()
    length = right_x - left_x
    res["xp"] -= left_x
    plt.figure(figsize=(8 * picScale, 1.3 * picScale))
    x = res["xp"]
    y = res["yp"]
    U = res["umag"]
    x, y = order_points_in_each_cluster(x=x, y=y)
    res["xp"] = x
    res["yp"] = y
    # plot all points
    # plt.plot(x, y, '-')
    curvature0 = None
    circle_x = np.array([])
    circle_y = np.array([])
    coords_up = np.array([])
    coords_down = np.array([])
    try:
        df, centers = find_df_centers(res, width=0.125)  # some points |y| < width
        # draw all centers
        for label, row in centers.iterrows():
            ind_label = df["label"] == label
            xx, yy = df[ind_label]["x"].values, df[ind_label]["y"].values
            circle_x, circle_y, x0, y0, nx0, ny0, curvature0 = plot_circle_with_curvature(xx, yy)
            # plt.plot(circle_x, circle_y, label=f'Circle with curvature {curvature0}')
            # plt.quiver(x0, y0, nx0, ny0, scale=10, color='red')
            # # plot approximate centers of circles
            # # plt.plot(circle_x, circle_y, 'c.', ms=2)
            # # plot sector between |y| < width corresponding
            # plt.plot(xx, yy, '-')

        # draw the second circle from right
        # plt.plot(centers["x"], centers["y"], 'r.')

        index, row = list(centers.index)[-2], centers.iloc[-2]
        ind_label = df["label"] == index
        xx, yy = df[ind_label]["x"].values, df[ind_label]["y"].values
        circle_x, circle_y, x0, y0, nx0, ny0, curvature0 = plot_circle_with_curvature(xx, yy)

        # find the start point
        df_cluster = df[(df["label"] == index)].copy(deep=True)  # take points of cluster "index"
        # take points of cluster "index" and above y>0
        df_ind = df[(df["label"] == index) & (df["y"] > 0)]
        # sort by x and take the last right element
        start = list(sorted(zip(df_ind["x"], df_ind["y"]), key=lambda x: x[0])[-1])
        logging.debug(f"start:{start}")
        # shift bubble to the beginning
        coords_up = give_coord(x, y, df, index, side="up")
        coords_down = give_coord(x, y, df, index, side="down")

        x_tip = start[0]
        U_tip = df[(df["x"] == start[0]) & (df["y"] == start[1])]["U"].iloc[0]
        circle_x -= x_tip
        x -= x_tip
        df["x"] -= x_tip
        df_cluster["x"] -= x_tip
        start[0] -= x_tip
        coords_up[:, 0] -= x_tip
        coords_down[:, 0] -= x_tip
        # choose index between point where y begins
        # decreasing and the tail of the bubble
        i_rmax_up = min(find_extremum(coords_up, True), np.argmin(coords_up[:, 0]))
        rmax_up = coords_up[i_rmax_up, 1]
        # choose index between point where y begins
        # decreasing and the tail of the bubble
        i_rmax_down = min(
            find_extremum(coords_down, False),
            np.argmin(coords_down[:, 0]),
        )
        rmax_down = coords_down[i_rmax_down, 1]
        logging.debug(
            f"up: {find_extremum(coords_up, True)} {np.argmin(coords_up[:,0])}",
        )
        logging.debug(
            f"down: {find_extremum(coords_down, False)} {np.argmin(coords_down[:,0])}",
        )
        logging.debug(
            f"rmax_up[{i_rmax_up}]={rmax_up} rmax_down[{i_rmax_down}]={rmax_down}",
        )
        # Output to the file
        outputs["t"].append(time)
        outputs["curvature_tip"].append(float(curvature0))
        outputs["U_tip"].append(U_tip)
        outputs["x_tip"].append(x_tip)
        outputs["rmax"].append(max(np.abs(rmax_down), np.abs(rmax_up)))

        # Draw

        width = np.abs(xmax - xmin)
        height = 1
        # plt.figure(figsize=(picScale1, (height / width) * picScale1))
        plt.plot(circle_x, circle_y, "c-", ms=2)
        # plt.plot(coords_up[::5,0], coords_up[::5,1], '-', lw=4)
        # plt.plot(coords_down[::5,0], coords_down[::5,1], '-', lw=4)
        plt.plot(x, y, ".", ms=2)  # all points
        plt.plot(df_cluster["x"], df_cluster["y"], "r.", ms=2)  # chosen only 1 cluster
        # plt.plot(df[:,0], df[:,1], 'r.', ms=2) #chosen for clustering |y| <0.1
        plt.quiver(start[0], start[1], nx0, ny0, scale=10, color='red')
        # plot approximate centers of circles
        # plt.plot(circle_x, circle_y, 'c.', ms=2)
        # plot sector between |y| < width corresponding
        # plt.plot(xx, yy, '-')
        # if rmax_up > rmax_down:
        plt.plot(
            [coords_up[i_rmax_up, 0]],
            [
                coords_up[i_rmax_up, 1],
            ],
            ".",
            ms=5,
        )  # Rmax point
        # else:
        plt.plot(
            [coords_down[i_rmax_down, 0]],
            [
                coords_down[i_rmax_down, 1],
            ],
            ".",
            ms=5,
        )  # Rmax point
        plt.plot([xmin, xmax], [-0.5, -0.5], c="0.55")
        plt.plot([xmin, xmax], [0.5, 0.5], c="0.55")
        plt.xlim(xmin, xmax)
        plt.ylim(-0.5, 0.5)
        # displaying the title
        plt.title(f"$t={time}$")
        # plt.axis('equal')
        plt.grid(True)
        plt.savefig(file[:-4] + "eps", bbox_inches="tight")
        plt.savefig(file[:-4] + "png", bbox_inches="tight")
        plt.cla()
    except Exception as e:
        logging.error({"info": f"Failed to process file: {file} at time: {time}", "cause": e})

with open("output_sliced_Rmax_U.json", "w") as f:
    json.dump(outputs, f)
