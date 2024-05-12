from __future__ import annotations

import glob
import json
import os
import re
from types import ModuleType

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import cos
from numpy import pi
from numpy import sin
from numpy import sqrt
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_lin_kernighan
from python_tsp.heuristics import solve_tsp_local_search
from python_tsp.heuristics import solve_tsp_record_to_record
from scipy import integrate
from scipy import interpolate
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.interpolate import sproot
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from timeout_decorator import timeout
from timeout_decorator import TimeoutError

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30
iter = 0
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels

Debug_plot = True
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
# matplotlib.rcParams['font.size'] = 25
matplotlib.pyplot.title(r"ABC123 vs $\mathrm{ABC123}^{123}$")

# set tight margins
plt.margins(0.015, tight=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return "DataFrame"
        elif isinstance(obj, pd.Series):
            return "Series"
        elif isinstance(obj, ModuleType):
            return "ModuleType"
        # print(obj)
        return json.JSONEncoder.default(self, obj)


def get_time(string):
    return float(re.findall(r"\d+\.\d+", string)[0])


def sort_names(image_files):
    file_names = [os.path.basename(string) for string in image_files]
    times = [(get_time(string), string) for string in file_names]
    times = sorted(times, key=lambda x: x[0])
    if times:
        print(f"Time: {times[0][0]} -- {times[-1][0]}")
    image_files = [t[1] for t in times]
    return image_files


#
# dX/dpsi=sin(psi)/Q
# dSigma/spsi = -cos(psi)/Q
# Q = sin(psi)/Sigma + s1*s2*X/l**2 - B
# X(0)=Sigma(0)=0


def fun_psi(psi, z, s1, s2, length, B):
    X, Sigma = z
    if psi < 1e-4:
        return [0, 2 / B]
    Q = sin(psi) / Sigma + s1 * s2 * X / length**2 - B
    return [sin(psi) / Q, -cos(psi) / Q]


#
# w'=f''=(1 + w**2)*(1/f + np.sqrt(1 + w**2)*(s1*s2*x/l**2 - B))
# f'=w


def fun_x(x, z, s1, s2, length, B):
    w, f = z
    return [-(1 + w**2) * (1 / f + np.sqrt(1 + w**2) * (-s1 * s2 * x / length**2 - B)), -w]


def volume(X, Sigma):
    y2 = Sigma**2
    Vd_comp = np.abs(np.pi * integrate.trapezoid(y2, x=X))
    return Vd_comp


def target_fun(X, Sigma, Vd):
    Vd_comp = volume(X, Sigma)
    Phi = np.abs(Vd_comp - Vd) / Vd
    # print(f'Vd_comp={Vd_comp} Vd={Vd} Phi={Phi}')
    return Phi


def shape_psi(B_, alpha, s1, s2, length):
    B = B_[0]
    X0 = 0
    Sigma0 = 0
    soln_psi = solve_ivp(
        fun_psi,
        (0, alpha),
        (X0, Sigma0),
        method="BDF",
        args=(s1, s2, length, B),
        min_step=1e-6,
        max_step=1e-3,
        rtol=1e-15,
        dense_output=True,
    )
    X_psi = soln_psi.y[0]
    Sigma_psi = soln_psi.y[1]
    return X_psi, Sigma_psi


def compute_shape_psi(B_, alpha, s1, s2, length, Vd):
    X_psi, Sigma_psi = shape_psi(B_, alpha, s1, s2, length)
    return target_fun(X_psi, Sigma_psi, Vd)


def shape_full(d_, B, s1, s2, length):
    X_psi, Sigma_psi = shape_psi([B], pi / 6, s1, s2, length)
    d = d_[0]
    X0 = -X_psi[-1]
    W0 = (Sigma_psi[-1] - Sigma_psi[-2]) / (X_psi[-1] - X_psi[-2])
    f0 = Sigma_psi[-1]
    print(f"X0={X0} f0={f0} W0={W0} d={d}, f()={fun_x(X0, (W0, f0), s1, s2, length, B)}")
    soln_x = solve_ivp(
        fun_x,
        (X0, d),
        (W0, f0),
        method="BDF",
        args=(s1, s2, length, B),
        min_step=1e-6,
        max_step=1e-3,
        rtol=1e-15,
        dense_output=True,
    )
    X_x = -soln_x.t
    # W_x = soln_x.y[0]
    Sigma_x = soln_x.y[1]
    # X = np.concatenate([X_psi, X_x])
    # Sigma = np.concatenate([Sigma_psi, Sigma_x])
    return X_psi, Sigma_psi, X_x, Sigma_x
    # return X, Sigma


def compute_shape_full(d_, B, s1, s2, length, Vd):
    X_psi, Sigma_psi, X_x, Sigma_x = shape_full(d_, B, s1, s2, length)
    X = np.concatenate([X_psi, X_x])
    Sigma = np.concatenate([Sigma_psi, Sigma_x])
    return target_fun(X, Sigma, Vd)


# s1 = -1 for pendant drop
# s2 = 1 for rhod - rhoa > 0


def pendant_drop(props, picScale, mode=None):
    Vd = props["Vd"]
    alpha = props["alpha"]
    s1 = props["s1"]
    s2 = props["s2"]
    diam = props["diam"]
    a = (3 * Vd / (4 * np.pi)) ** (1 / 3)
    props["a"] = a
    d = a * props["d/diam"]
    length = np.sqrt(props["sigma"] / ((props["rho1"] - props["rho2"]) * props["grav"]))
    iBo = (length / a) ** 2
    B = 2 / (a * (4 / (2 + cos(alpha) ** 3 - 3 * cos(alpha))) ** (1 / 3))
    # B *= 0.9
    print(f"Ba_guess={B*a}, d/diam={d/diam}, a={a}, l/diam={length/diam}, a/Bo={iBo*a}")
    if not mode:
        # res = minimize(
        #     compute_shape_full,
        #     x0=[d],
        #     args=(B, s1, s2, length, Vd),
        #     method='L-BFGS-B', jac=None,
        #     options={'gtol': 1e-5, 'disp': Debug_plot}
        # )
        # d = res.x[0]
        print(f"Ba_res={B} d_res/diam={d/diam}")
        # X, Sigma = shape_full([d], B, s1, s2, length)
        X_psi, Sigma_psi, X_x, Sigma_x = shape_full([d], B, s1, s2, length)
        width = np.abs(X_x.min())
        height = np.abs(max(Sigma_psi.max(), Sigma_x.max()))
        plt.figure(figsize=(picScale, (height / width) * picScale))
        plt.plot(X_psi / diam, Sigma_psi / diam, ".-")
        plt.plot(X_x / diam, Sigma_x / diam, ".-")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.savefig("Sigma_X.png", bbox_inches="tight")
        plt.cla()
    else:
        res = minimize(
            compute_shape_psi,
            x0=[B],
            args=(alpha, s1, s2, length, Vd),
            method="L-BFGS-B",
            jac=None,
            options={"gtol": 1e-6, "disp": Debug_plot},
        )
        B = res.x[0]
        X, Sigma = shape_psi([B], alpha, s1, s2, length)

        plt.figure(figsize=(8 * picScale, 1.3 * picScale))
        plt.plot(X / diam, Sigma / diam, ".")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.savefig("Sigma_X.png", bbox_inches="tight")
        plt.cla()


def periodic_linspace(x_min, x_max, num=100, endpoint=True):
    if x_min < x_max:
        return np.linspace(x_min, x_max, num=num, endpoint=endpoint)
    else:
        if endpoint:
            return np.concatenate((
                np.linspace(0, x_max, num=num//2, endpoint=True),
                np.linspace(x_min, 1, num=num//2, endpoint=True),
            ))
        else:
            return np.concatenate((
                np.linspace(0, x_max, num=num//2, endpoint=False),
                np.linspace(x_min, 1, num=num//2, endpoint=False),
            ))


def func_curvature(pars, xx, yy):
    xc, yc, R = pars
    return sum([((xi - xc) ** 2 + (yi - yc) ** 2 - R**2)**2 for xi, yi in zip(xx, yy)])


def jac_curvature(pars, xx, yy):
    xc, yc, R = pars
    df_dxc = -4 * sum([
        (xi - xc) * ((xi - xc) ** 2 + (yi - yc) ** 2 - R**2)
        for xi, yi in zip(xx, yy)
    ])
    df_dyc = -4 * sum([
        (yi - yc) * ((xi - xc) ** 2 + (yi - yc) ** 2 - R**2)
        for xi, yi in zip(xx, yy)
    ])
    df_dR = -4 * R * sum([((xi - xc) ** 2 + (yi - yc) ** 2 - R**2) for xi, yi in zip(xx, yy)])
    return df_dxc, df_dyc, df_dR


def find_df_centers(res, width=0.1):
    x = res["xp"]
    y = res["yp"]
    U = res["umag"]
    points = np.c_[x, y]

    ind = np.abs(y) < width
    points = points[ind]
    U = U[ind]
    clustering = DBSCAN(eps=0.1, min_samples=2).fit(points)
    df = pd.DataFrame({
        "x": points[:, 0],
        "y": points[:, 1],
        "label": clustering.labels_,
        "U": U,
    })
    centers = df.groupby("label").mean()  # label becomes as an index
    centers.sort_values(by="x", inplace=True)
    print(f"centers: {centers}")
    return df, centers


def compute_curvature(index, row, df, a):
    phi = np.linspace(0, 2 * np.pi, 1000)
    center_x, _ = row["x"], row["y"]
    df_label = df[df["label"] == index]
    result = minimize(
        func_curvature,
        x0=[center_x, 0, 25 * a],
        args=(df_label["x"].values, df_label["y"].values),
        method="L-BFGS-B",
        jac=None,
        options={"gtol": 1e-7, "disp": Debug_plot},
    )
    result_curvature = 2 / result.x[2]
    print(
        f"RES: center_x={result.x[0]} center_y={result.x[1]} R={result.x[2]}, "
        f"R={result.x[2]} curvature={result_curvature} using {df_label.shape[0]} points",
    )
    x_circle_result, y_circle_result = result.x[0] + result.x[2] * \
        np.cos(phi), result.x[1] + result.x[2] * np.sin(phi)
    return x_circle_result, y_circle_result, result_curvature


def compute_curvature_from_ordered_points(tck, specific_u):
    # Evaluate first derivatives at the specific parameter u
    dx_du, dy_du, _, _ = interpolate.splev(specific_u, tck, der=1)
    d2x_du2, d2y_du2, _, _ = interpolate.splev(specific_u, tck, der=2)

    # Compute curvature using the formula: curvature =
    # |dx_du * d2y_du2 - dy_du * d2x_du2| / (dx_du^2 + dy_du^2)^(3/2)
    curvature = (dx_du * d2y_du2 - dy_du * d2x_du2) / (dx_du**2 + dy_du**2)**(3/2)

    return curvature


def find_u_for_parametric_curve(tck, specific_u, target_distance=0.1, bounds=(0, 1)):
    # Define the speed function
    def speed_fun(u_point):
        u_point %= 1
        x_d, y_d, _, _ = interpolate.splev(u_point, tck, der=1)
        return np.sqrt(x_d**2 + y_d**2)

    # Define a function to compute the distance between a point u0 and a point on the spline curve
    def distance_to_spline(u_point):
        u_point %= 1
        u_left = min(u_point, specific_u)
        u_right = max(u_point, specific_u)
        length0, _ = quad(speed_fun, u_left, u_right)
        length1, _ = quad(speed_fun, 0, u_left)
        length2, _ = quad(speed_fun, u_right, 1)
        return min(length0, length1 + length2)
    # Find the parameter values u_left and u_right
    u = minimize_scalar(
        lambda u: np.abs(
            distance_to_spline(
                u,
            ) - target_distance,
        ), bounds=bounds, method='bounded',
    ).x % 1
    return u


def compute_curvature_from_ordered_points_averaged(tck, specific_u):
    # Find the parameter values u_left and u_right
    u_left = find_u_for_parametric_curve(
        tck, specific_u=specific_u, target_distance=0.15, bounds=(specific_u-0.1, specific_u),
    )
    u_right = find_u_for_parametric_curve(
        tck, specific_u=specific_u, target_distance=0.15, bounds=(specific_u, specific_u+0.1),
    )
    print("u_left:", u_left, "u_right:", u_right)
    x_left, y_left, _, _ = interpolate.splev(u_left, tck)
    x_right, y_right, _, _ = interpolate.splev(u_right, tck)
    print("x_left:", x_left, "y_left:", y_left)
    print("x_right", x_right, "y_right:", y_right)
    u = periodic_linspace(u_left, u_right, num=100, endpoint=True)
    new_xx, new_yy, _, _ = interpolate.splev(u, tck)
    center_x = new_xx.mean()
    result = minimize(
        func_curvature,
        x0=[center_x, 0, 0.25],  # the initial guess for xc, yc, radius
        args=(new_xx, new_yy),
        method="L-BFGS-B",
        jac=jac_curvature,
        options={"gtol": 1e-7, "disp": Debug_plot, "maxiter": 1000, "eps": 1e-08},
    )
    xc = result.x[0]
    yc = result.x[1]
    rc = abs(result.x[2])
    result_curvature = 1 / rc
    print(
        f"RES: center_x={xc} center_y={yc} R={rc}, curvature={result_curvature} using {len(new_xx)} points.\n"
        f"Specific u: {specific_u}.\n"
        f"New_xx: [{new_xx.min()}, {new_xx.max()}].\n"
        f"New_yy: [{new_yy.min()}, {new_yy.max()}].",
    )
    return xc, yc, rc, result_curvature, x_left, y_left, x_right, y_right


def compute_normals_from_ordered_points(tck, specific_u):
    # Compute first derivative of the spline at the specific parameter u
    dx_du, dy_du, _, _ = interpolate.splev(specific_u, tck, der=1)

    # Compute normals
    normal_x = -dy_du
    normal_y = dx_du
    normal_len = sqrt(normal_x**2 + normal_y**2)
    return normal_x/normal_len, normal_y/normal_len


def plot_circle(x0, y0, curvature, nx0, ny0):
    # Calculate the radius from the curvature
    radius = 1 / np.abs(curvature)
    x_center = x0 + radius * nx0
    y_center = y0 + radius * ny0

    # Generate angles
    angles = np.linspace(0, 2 * np.pi, 100)

    # Calculate circle coordinates
    circle_x = x_center + radius * np.cos(angles)
    circle_y = y_center + radius * np.sin(angles)
    return circle_x, circle_y


def plot_circle_with_curvature(xx, yy, ux, umag, label, smooth_parameter=0.001):
    res = []
    characteristic_size = np.ptp(xx)
    smooth_parameter *= characteristic_size
    tck, u = interpolate.splprep([xx, yy, ux, umag], s=smooth_parameter, per=True)
    u = np.linspace(0, 1, 1000)
    new_xx, new_yy, new_ux, new_umag = interpolate.splev(u, tck)
    # find coordinates of cross-sections with y=0
    _, u_y_zeros, _, _ = sproot(tck)
    if len(u_y_zeros) == 0:
        return [{
            "label": label,
            "tck": tck,
            "new_xx": new_xx,
            "new_yy": new_yy,
            "new_ux": new_ux,
            "new_umag": new_umag,
        }]

    root_x, root_y, root_ux, root_umag = interpolate.splev(u_y_zeros, tck)
    for i in range(len(root_x)):
        specific_u = u_y_zeros[i]
        x0, y0, u0, umag0 = root_x[i], root_y[i], root_ux[i], root_umag[i]
        print(
            "root: ", i, "x0:", x0, "y0:", y0, "u0:", u0,
            "umag0:", umag0, "specific_u:", specific_u,
        )
        if False:
            curvature0 = compute_curvature_from_ordered_points(tck, specific_u)
            nx0, ny0 = compute_normals_from_ordered_points(tck, specific_u)
            # Determine concavity based on curvature
            concave_inward = curvature0 > 0
            # Adjust the direction of the normal vector based on concavity
            nx0 = nx0 if concave_inward else -nx0
            ny0 = ny0 if concave_inward else -ny0
        else:
            xc, yc, rc, curvature0, x_left, y_left, x_right, y_right = compute_curvature_from_ordered_points_averaged(
                tck, specific_u,
            )
            v = np.array([xc - x0, yc - y0])
            nx0, ny0 = v / (np.linalg.norm(v) + 1e-16)
        print("+++curvature0:", curvature0, "nx0:", nx0, "ny0:", ny0)
        circle_x, circle_y = plot_circle(x0, y0, curvature0, nx0, ny0)
        res.append({
            "label": label,
            "tck": tck,
            "u": specific_u,
            "x0": x0,
            "y0": y0,
            "u0": u0,
            "umag0": umag0,
            "curvature0": curvature0,
            "nx0": nx0,
            "ny0": ny0,
            "circle_x": circle_x,
            "circle_y": circle_y,
            "ux": root_ux[i],
            "umag": root_umag[i],
            "new_xx": new_xx,
            "new_yy": new_yy,
            "new_ux": new_ux,
            "new_umag": new_umag,
            "x_left": x_left,
            "y_left": y_left,
            "x_right": x_right,
            "y_right": y_right,
        })
    return sorted(res, key=lambda v: v["x0"], reverse=True)


def distance(p1, p2):
    """
    This function computes the distance between 2 points defined by
     P1 = (x1,y1) and P2 = (x2,y2)
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def optimized_path(coords, start=None):
    """
    This function finds the nearest point to a point
    coords should be a list in this format coords = [ [x1, y1], [x2, y2] , ...]
    """
    if not isinstance(coords, list):
        coords = [[x[0], x[1]] for x in coords]

    if start is None:
        start = coords[0]
    start = min(coords, key=lambda x: distance(start, x))

    pass_by = coords
    path = [start]
    pass_by.remove(start)
    while pass_by:
        nearest = min(pass_by, key=lambda x: distance(path[-1], x))
        path.append(nearest)
        pass_by.remove(nearest)
    return np.asarray(path)


def distance_between_points(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def total_path_length(x, y):
    return np.sum(distance_between_points(x[:-1], y[:-1], x[1:], y[1:]))


def clusterize_points(res, width=0.05):
    points = np.c_[res["xp"], res["yp"]]
    clustering = DBSCAN(eps=width, min_samples=2).fit(points)
    df = pd.DataFrame({
        "xp": points[:, 0],
        "yp": points[:, 1],
        "label": clustering.labels_,
        "umag": res["umag"],
        "ux": res["ux"],
    })
    res["labels"] = np.unique(clustering.labels_)
    res["df"] = df.sort_values(by="label", kind="stable")
    res["logger"].info({
        "info": "Points are clustered using DBSCAN",
        "labels": res["labels"],
    })


def shift_to_xmin(res):
    left_x = res["xp"].min()
    right_x = res["xp"].max()
    length = right_x - left_x
    res["xp"] -= left_x
    res["length"] = length


# Timeout set to 5 minutes (300 seconds)
def order_points_using_tsp(x, y, permutation0=None):
    sources = np.stack((x, y), axis=1)
    destinations = np.stack((x, y), axis=1)
    distance_matrix = euclidean_distance_matrix(sources, destinations)
    # permutation, _ = solve_tsp_record_to_record(distance_matrix, x0=permutation0)
    permutation, _ = solve_tsp_lin_kernighan(distance_matrix, x0=permutation0)
    # permutation, _ = solve_tsp_local_search(distance_matrix, x0=permutation0, log_file="tsp_log")
    return permutation


def order_points(x, y, permutation0=None, max_retries=5, timeout_seconds=300) -> list[int]:
    for n in range(max_retries):
        try:
            print("Trying to order points. Attempt:", n+1)
            result = timeout(timeout_seconds)(order_points_using_tsp)(x, y, permutation0)
            print("Successfully order with attempt:", n+1)
            return result
        except TimeoutError:
            permutation0 = None
            print(f"Function took too long to execute. Retrying... {n+1}/{max_retries}")
    return np.arange(0, len(x))


def order_points_in_each_cluster(res, max_retries=5, timeout_seconds=300) -> None:
    res["logger"].info("Points ordering")
    df = res["df"]
    permutation_total = []
    for label in res["labels"]:
        n_points = df["xp"][df["label"] == label].size
        res["logger"].info({"Ordering label": label, "n_points": n_points})
        index_label = df["label"] == label
        permutation0 = np.arange(0, n_points).tolist()
        # permutation0 = np.random.permutation(n_points).tolist()
        permutation = order_points(
            x=df["xp"][index_label].values,
            y=df["yp"][index_label].values,
            permutation0=permutation0,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
        permutation = [p + len(permutation_total) for p in permutation]
        permutation_total = permutation_total + permutation
    res["df"] = df.iloc[permutation_total].reset_index(drop=True)
    for key in ["xp", "yp", "ux", "umag", "label"]:
        res[key] = res["df"][key].values
    res["logger"].info("Ordered successfully")


# calculate Rmax
def calculate_rmax(second_tip, logging):
    tck = second_tip["tck"]
    u0 = second_tip["u"]
    u_left = find_u_for_parametric_curve(
        tck, specific_u=u0, target_distance=1, bounds=(u0-0.3, u0),
    )
    u_right = find_u_for_parametric_curve(
        tck, specific_u=u0, target_distance=1, bounds=(u0, u0+0.3),
    )
    print("u_left:", u_left, "u_right:", u_right)
    x_left, y_left, _, _ = interpolate.splev(u_left, tck)
    x_right, y_right, _, _ = interpolate.splev(u_right, tck)
    print("x_left:", x_left, "y_left:", y_left)
    print("x_right", x_right, "y_right:", y_right)
    # u = periodic_linspace(u_left, u_right, num=100, endpoint=True)

    logging.debug(f"find peaks in range [{u_left}, {second_tip['u']}]")

    u_left_arr = np.linspace(u_left, u0, 100)
    xx_left, yy_left, _, _ = interpolate.splev(u_left_arr, second_tip["tck"])
    i_max = np.abs(yy_left).argmax()

    # peaks_left, _ = find_peaks(yy_left, height=0.1)  #, prominence=0.01
    # xx_peak_coords_left = xx_left[peaks_left] if len(peaks_left) else np.array([xx_left[i_max]])
    # yy_peak_coords_left = yy_left[peaks_left] if len(peaks_left) else np.array([yy_left[i_max]])
    # logging.debug(f"find_peaks left peaks={peaks_left} xx={xx_peak_coords_left} yy={yy_peak_coords_left}")

    xx_peak_coords_left = np.array([xx_left[i_max]])
    yy_peak_coords_left = np.array([yy_left[i_max]])
    logging.debug(f"find_peaks left xx={xx_peak_coords_left} yy={yy_peak_coords_left}")

    logging.debug(f"find peaks in range [{u0}, {u_right}]")
    u_right_arr = np.linspace(u0, u_right, 100)
    xx_right, yy_right, _, _ = interpolate.splev(u_right_arr, second_tip["tck"])
    i_max = np.abs(yy_right).argmax()
    # peaks_right, _ = find_peaks(-yy_right, height=0.1)  #, prominence=0.01
    # xx_peak_coords_right = xx_right[peaks_right] if len(peaks_right) else np.array([xx_right[i_min]])
    # yy_peak_coords_right = yy_right[peaks_right] if len(peaks_right) else np.array([yy_right[i_min]])
    # logging.debug(f"find_peaks right peaks={peaks_right} xx={xx_peak_coords_right} yy={yy_peak_coords_right}")
    xx_peak_coords_right = np.array([xx_right[i_max]])
    yy_peak_coords_right = np.array([yy_right[i_max]])
    logging.debug(f"find_peaks right xx={xx_peak_coords_right} yy={yy_peak_coords_right}")
    if len(yy_peak_coords_left) == 0 and len(yy_peak_coords_right) == 0:
        logging.error("No peaks found")
        return 0
    elif len(yy_peak_coords_left) == 0:
        rmax = 0.5 * np.abs(yy_peak_coords_right[0])
    elif len(yy_peak_coords_right) == 0:
        rmax = 0.5 * np.abs(yy_peak_coords_left[0])
    else:
        rmax = 0.5 * (np.abs(yy_peak_coords_left[0]) + np.abs(yy_peak_coords_right[0]))

    logging.debug(f"rmax={rmax}")
    second_tip["rmax"] = rmax
    second_tip["xx_peak_left"] = xx_peak_coords_left
    second_tip["yy_peak_left"] = yy_peak_coords_left
    second_tip["xx_peak_right"] = xx_peak_coords_right
    second_tip["yy_peak_right"] = yy_peak_coords_right
    second_tip["xx_left"] = xx_left[0]
    second_tip["yy_left"] = yy_left[0]
    second_tip["xx_right"] = xx_right[-1]
    second_tip["yy_right"] = yy_right[-1]
    return rmax


def uniform_interpolation(x, y, xn, yn):
    N = 1000
    print(x.min(), xn.min())
    xmin = max(x.min(), xn.min())
    xmax = min(x.max(), xn.max())
    xx = np.linspace(xmin, xmax, N)
    f = interp1d(x, y)
    yy = f(xx)

    f = interp1d(xn, yn)
    yyn = f(xx)
    # return LA.norm(yy - yyn, ord=1)*(xmax - xmin)
    return np.abs(yy - yyn).sum()


def get_closest_ind(array, values):
    # make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = (idxs == len(array)) | (
        np.fabs(values - array[np.maximum(idxs - 1, 0)])
        < np.fabs(values - array[np.minimum(idxs, len(array) - 1)])
    )

    try:
        idxs[prev_idx_is_less] -= 1
        return idxs[0]
    except Exception:
        return idxs


def fit_curve_err(par, props, coords):
    tail_y, B0, length = par
    diam = props["diam"]
    d = diam * props["d/diam"]  # TODO: be careful
    global iter
    print(f"fit_curve_err: par: {par}, d/diam: {d/diam}")

    coords_filtered = coords[
        (coords[:, 0] > -np.abs(d) / diam) & (coords[:, 0] < 0)
    ]  # take points until d

    id = get_closest_ind(coords_filtered[:, 0], -np.abs(d) / diam)
    point_left = coords_filtered[id]
    print(f"Closest left point/diam: {point_left/diam}")
    # print(f'coords: {coords}')
    # print(f'coords_filtered: {coords_filtered}')

    # B = props['B']
    s1 = props["s1"]
    s2 = props["s2"]
    X_psi, Sigma_psi, X_x, Sigma_x = shape_full([np.abs(d)], B0, s1, s2, length)
    # width = np.abs(X_x.min())
    # height = np.abs(max(Sigma_psi.max(), Sigma_x.max()))
    # plt.figure(figsize=(picScale, (height/width)*picScale))
    # plt.plot(X_psi/diam, Sigma_psi/diam)
    # plt.plot(X_x/diam, Sigma_x/diam)
    # plt.grid(True)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.axis('equal')
    # plt.savefig(f'fit_curve_err_{iter}.png', bbox_inches='tight')
    # plt.cla()

    iter += 1
    # print(X_psi, Sigma_psi, X_x, Sigma_x)
    print(f"X_psi: {X_psi.min()} {X_psi.max()}")
    print(f"X_x: {X_x.min()} {X_x.max()}")
    coords_filtered1 = coords_filtered[coords_filtered[:, 0] > X_psi.min() / diam]
    coords_filtered2 = coords_filtered[coords_filtered[:, 0] <= X_psi.min() / diam]
    # print(f'coords_filtered1: {coords_filtered1}')
    # print(f'coords_filtered2: {coords_filtered2}')
    err1 = err2 = 0
    if len(coords_filtered1):
        err1 = uniform_interpolation(
            X_psi / diam,
            Sigma_psi / diam,
            coords_filtered1[:, 0],
            coords_filtered1[:, 1],
        )
    if len(coords_filtered2):
        err2 = uniform_interpolation(
            X_x / diam,
            Sigma_x / diam,
            coords_filtered2[:, 0],
            coords_filtered2[:, 1],
        )
    return err1 + err2 + 10 * distance(point_left, [props["d/diam"], tail_y / diam])


def fit_curve(par0, props, coords):
    res = dict({"x": par0})
    res = minimize(
        fit_curve_err,
        x0=par0,
        args=(props, coords),
        method="Nelder-Mead",
    )
    # options={'gtol': 1e-4, 'disp': Debug_plot})
    d = props["d"]
    print(res["x"])
    tail_y, B0, length = res["x"]
    X_psi, Sigma_psi, X_x, Sigma_x = shape_full([np.abs(d)], B0, props["s1"], props["s2"], length)
    props["d"] = np.abs(d)
    props["d/diam"] = np.abs(d) / props["diam"]
    props["l"] = length
    diam = props["diam"]
    plt.plot(X_psi / diam, Sigma_psi / diam, ".-")
    plt.plot(X_x / diam, Sigma_x / diam, ".-")
    plt.grid(True)
    plt.savefig("shape_full.eps", bbox_inches="tight")
    plt.cla()
    return X_psi / diam, Sigma_psi / diam, X_x / diam, Sigma_x / diam, props


if __name__ == "__main__":
    csvPattern = "slice_t=*.csv"
    xmin = -4
    xmax = 4
    ymax = 1.5
    picScale = 4
    picScale1 = 16

    props = {}
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
    df = pd.read_csv(
        "/Users/weugene/Desktop/toDelete/points_drop.csv",
        sep=",",
        usecols=["Points_0", "Points_1"],
    )
    df = df[df["Points_1"] > 0]
    left_x = df["Points_0"].min()
    df["Points_0"] -= left_x
    df = df.sort_values("Points_0")
    df = df.reset_index(drop=True)
    # print(df)
    props["Vd"] = (
        volume(df["Points_0"].values, df["Points_1"].values) * props["diam"] ** 3
    )  # 0.2179e-9
    print(f"props={props}")

    pendant_drop(props, picScale1)

    csvnames = glob.glob(f"./{csvPattern}", recursive=False)
    csvnames = sort_names(csvnames)
    print("Found pvd files in:", csvnames)

    dpdx = (0.428276 - 0.510173) / (3.95884 - 5.19025)
    # props['grav'] = dpdx/(props['rho1'] - props['rho2'])
    props["l"] = np.sqrt(
        props["sigma"] / ((props["rho1"] - props["rho2"]) * props["grav"]),
    )  # ???? TODO: see here

    for ifile, file in enumerate(csvnames):
        time = get_time(file)
        # if ifile != 2:
        #     continue
        print(f"file: {file} time: {time}")
        res = pd.read_csv(
            file,
            sep=",",
            usecols=["Points_0", "Points_1", "u.x_0", "u.x_1", "u.x_2", "u.x_Magnitude"],
        )
        left_x = res["Points_0"].min()
        right_x = res["Points_0"].max()
        length = right_x - left_x
        res["Points_0"] -= left_x
        plt.figure(figsize=(8 * picScale, 1.3 * picScale))
        x = res["xp"]
        y = res["yp"]
        df, centers = find_df_centers(res, width=0.1)
        # draw all centers
        # for index, row in centers.iterrows():
        #     x_circle, y_circle, curvature = compute_curvature(index, row, df, props["a"])
        #     plt.plot(x_circle, y_circle, 'c.', ms=2)
        # draw the second circle from right
        index, row = list(centers.index)[-2], centers.iloc[-2]
        print("row", row, "index", index)
        x_circle, y_circle, curvature = compute_curvature(index, row, df, props["a"])

        # find the start point
        df_cluster = df[(df["label"] == index)]
        df_ind = df[(df["label"] == index) & (df["y"] > 0)]
        start = list(sorted(zip(df_ind["x"].values, df_ind["y"].values), key=lambda x: x[0])[-1])
        print(f"start:{start}")
        # shift bubble to the beginning
        x_tip = start[0]
        x_circle -= x_tip
        x -= x_tip
        df["x"] -= x_tip
        df_cluster["x"] -= x_tip
        start[0] -= x_tip

        coords = np.c_[x, y]
        coords = coords[coords[:, 1] > 0]
        clustering = DBSCAN(eps=0.02, min_samples=2).fit(coords)
        print(f"Found N={clustering} clusters")
        df_compare = pd.DataFrame(
            {"x": coords[:, 0], "y": coords[:, 1], "label": clustering.labels_},
        )
        # print(f'df: {df_compare}')
        label = df_compare[(df_compare["x"] == start[0]) & (df_compare["y"] == start[1])][
            "label"
        ].values[0]
        print(f"label: {label}")
        ind = df_compare["label"] == label
        coords = np.c_[df_compare["x"][ind].values, df_compare["y"][ind].values]
        coords = optimized_path(coords, start)
        # print(f'coords = {coords}')
        # Find

        props["d/diam"] = 1  # 1.3#0.8801273617937209  #variable parameter 2.2
        props["tail_y/diam"] = 0.2
        props["l"] = 2.54107886e-04  # 0.00020038 #0.00020039#0.00019
        # props['Vd'] = 8e-1 #3.642e-11
        d0 = props["d/diam"] * props["diam"]
        tail_y = props["tail_y/diam"] * props["diam"]
        B0 = curvature / props["diam"]
        l0 = props["l"]
        props["B"] = curvature / props["diam"]
        props["d"] = props["d/diam"] * props["diam"]
        X_psi_theor, Sigma_psi_theor, X_x_theor, Sigma_x_theor, props = fit_curve(
            (tail_y, B0, l0),
            props,
            coords,
        )
        # print([a for a in zip(X_psi_theor, Sigma_psi_theor)])
        # print([a for a in zip(X_x_theor, Sigma_x_theor)])

        print(f"props[d/diam] {props['d/diam']}")

        df = df.values
        df_cluster = df_cluster.values

        width = np.abs(xmax - xmin)
        height = 1
        plt.figure(figsize=(picScale1, (height / width) * picScale1))
        plt.plot(x_circle, y_circle, "c.", ms=2)
        # plt.plot(coords[::5,0], coords[::5,1], '-', lw=4)
        plt.plot(x, y, ".", ms=2)  # all points
        plt.plot(X_psi_theor, Sigma_psi_theor, "y-")
        plt.plot(X_x_theor, Sigma_x_theor, "y-")
        plt.plot(X_psi_theor, -Sigma_psi_theor, "y-")
        plt.plot(X_x_theor, -Sigma_x_theor, "y-")
        plt.plot(df_cluster[:, 0], df_cluster[:, 1], "r.", ms=2)  # chosen only 1 cluster
        # plt.plot(df[:,0], df[:,1], 'r.', ms=2) #chosen for clustering |y| <0.1

        plt.plot([xmin, xmax], [-0.5, -0.5], c="0.55")
        plt.plot([xmin, xmax], [0.5, 0.5], c="0.55")
        plt.xlim(xmin, xmax)
        plt.ylim(-0.5, 0.5)
        # plt.axis('equal')
        plt.grid(True)
        plt.savefig(file[:-3] + "eps", bbox_inches="tight")
        plt.savefig(file[:-3] + "png", bbox_inches="tight")
        plt.cla()
