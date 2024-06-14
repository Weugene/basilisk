from typing import Iterable

import numpy as np
import scipy.integrate as scp_integrate
import scipy.interpolate as scp_interpolate
import scipy.optimize as scpo
from matplotlib import pyplot as plt
from numpy import ndarray

# s1 = -1 for pendant drop
# s2 = 1 for rho_d - rho_a > 0
iter = 0

class Config:
    def __init__(self, props: dict):
        self.s1: int = props.get("s1", -1)
        self.s2: int = props.get("s2", 1)
        self.B: float = 2*props.get("curvature", 0)
        self.psi1: float = 0
        self.psi2: float = props.get("psi2", np.pi/2)
        second_tip = props.get("second_tip", {})
        coords_left = np.c_[second_tip["xx_left"], second_tip["yy_left"]]
        self.coords_left = coords_left
        coords_right = np.c_[second_tip["xx_right"], second_tip["yy_right"]]
        self.coords_right = coords_right
        self.x0 = second_tip["x0"]
        self.y0 = second_tip["y0"]
        print(f"B={self.B}, x0={self.x0}, y0={self.y0}")


# x = X(psi), sigma = Sigma(psi)
# dX/dpsi = sin(psi)/Q
# dSigma/dpsi = -cos(psi)/Q
# Q = sin(psi)/Sigma + s1*s2*X/l**2 - B
# dX/dpsi(psi=0) = 0
# dSigma/dpsi(psi=0) = 2/B
# X(0)=Sigma(0)=0


def fun_psi(psi: float, z: tuple[float, float], length: float, config: Config) -> list[float, float]:
    X, Sigma = z
    if psi < 1e-4:
        return [0, 2 / config.B]
    Q = np.sin(psi) / Sigma + config.s1 * config.s2 * X / length**2 - config.B
    return [
        np.sin(psi) / Q,
        -np.cos(psi) / Q
    ]


#
# w'=f''=(1 + w**2)*(1/f + np.sqrt(1 + w**2)*(s1*s2*x/l**2 - B))
# f'=w

# def fun_x(x: float, z: tuple[float, float], config: Config) -> list[float, float]:
#     w, f = z
#     return [
#         (1 + w**2) * (1 / f + np.sqrt(1 + w**2) * (config.s1 * config.s2 * x / config.length**2 - config.B)),
#         w
#     ]


# def volume(X: ndarray, Sigma: ndarray) -> float:
#     y2 = Sigma**2
#     Vd_comp = np.abs(np.pi * scp_integrate.trapezoid(y2, x=X))
#     return Vd_comp
#
#
# def target_fun(X: ndarray, Sigma: ndarray, Vd: float) -> float:
#     Vd_comp = volume(X, Sigma)
#     Phi = np.abs(Vd_comp - Vd) / Vd
#     # print(f'Vd_comp={Vd_comp} Vd={Vd} Phi={Phi}')
#     return Phi


def shape_psi(length: float, config: Config) -> tuple[ndarray, ndarray]:
    X0 = 0
    Sigma0 = 0
    solution_psi = scp_integrate.solve_ivp(
        fun=fun_psi,
        t_span=(config.psi1, config.psi2),
        y0=(X0, Sigma0),
        method="BDF",
        args=[length, config],
        min_step=1e-6,
        max_step=1e-3,
        rtol=1e-15,
        dense_output=True,
    )
    X_psi = solution_psi.y[0]
    Sigma_psi = solution_psi.y[1]
    return X_psi, Sigma_psi


# def compute_shape_psi(Vd, config: Config) -> float:
#     X_psi, Sigma_psi = shape_psi(config)
#     return target_fun(X_psi, Sigma_psi, Vd)

def uniform_interpolation(x, y, xn, yn):
    N = 1000
    print(x.min(), xn.min())
    x_min = max(x.min(), xn.min())
    x_max = min(x.max(), xn.max())
    xx = np.linspace(x_min, x_max, N)
    f = scp_interpolate.interp1d(x, y)
    yy = f(xx)

    f = scp_interpolate.interp1d(xn, yn)
    yyn = f(xx)
    # return LA.norm(yy - yyn, ord=1)*(xmax - xmin)
    return np.abs((yy - yyn)).sum()


def fit_curve_err(par, config: Config, part: str):
    length, = par
    print(f"fit_curve_err: par: {par}, config: {config}")
    if part == "left":
        coords = config.coords_left
    else:
        coords = config.coords_right
    X_psi, Sigma_psi = shape_psi(length, config)
    err = uniform_interpolation(
        X_psi,
        Sigma_psi,
        coords[:, 0],
        coords[:, 1],
    )
    return err
    # global iter
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
    # iter += 1

    # print(X_psi, Sigma_psi, X_x, Sigma_x)
    print(f"X_psi: {X_psi.min()} {X_psi.max()}")
    print(f"X_x: {X_x.min()} {X_x.max()}")
    stitch_x = X_psi.min()
    coords_filtered1 = coords_filtered[coords_filtered[:, 0] > stitch_x]
    coords_filtered2 = coords_filtered[coords_filtered[:, 0] <= stitch_x]
    # print(f'coords_filtered1: {coords_filtered1}')
    # print(f'coords_filtered2: {coords_filtered2}')
    err1 = err2 = 0
    if len(coords_filtered1):
        err1 = uniform_interpolation(
            X_psi,
            Sigma_psi,
            coords_filtered1[:, 0],
            coords_filtered1[:, 1],
            )
    if len(coords_filtered2):
        err2 = uniform_interpolation(
            X_x,
            Sigma_x,
            coords_filtered2[:, 0],
            coords_filtered2[:, 1],
            )
    return err1 + err2


# unknown variable is capillary length `length`
def fit_curve(length_guess, config: Config):
    res = dict({"x": [length_guess]})
    res = scpo.minimize(
        fit_curve_err,
        x0=[length_guess],
        args=(config, "left"),
        method="Nelder-Mead",
    )
    length_left, = res.x
    print(f"length_left={length_left}")
    X_psi_left, Sigma_psi_left = shape_psi(length_left, config)

    res = scpo.minimize(
        fit_curve_err,
        x0=[length_guess],
        args=[config, "right"],
        method="Nelder-Mead",
    )
    length_right, = res.x
    print(f"length_left={length_right}")
    X_psi_right, Sigma_psi_right = shape_psi(length_right, config)

    plt.plot(X_psi_left, Sigma_psi_left, ".-")
    plt.plot(X_psi_right, Sigma_psi_right, "-")
    plt.grid(True)
    plt.savefig("shape_full.eps", bbox_inches="tight")
    plt.cla()
    X_psi = np.concatenate([X_psi_left, X_psi_right])
    Sigma_psi = np.concatenate([Sigma_psi_left, Sigma_psi_right])
    return X_psi, Sigma_psi





# def shape_full(d_: Iterable, config: Config) -> tuple[ndarray, ndarray, ndarray, ndarray]:
#     X_psi, Sigma_psi = shape_psi(config)
#     d, = d_
#     X0 = -X_psi[-1]
#     W0 = (Sigma_psi[-1] - Sigma_psi[-2]) / (X_psi[-1] - X_psi[-2])
#     f0 = Sigma_psi[-1]
#     print(f"X0={X0} f0={f0} W0={W0} d={d}, f()={fun_x(X0, (W0, f0), config)}")
#     solution_x = scp_integrate.solve_ivp(
#         fun=fun_x,
#         t_span=(X0, d),
#         y0=(W0, f0),
#         method="BDF",
#         args=[config],
#         min_step=1e-6,
#         max_step=1e-3,
#         rtol=1e-15,
#         dense_output=True,
#     )
#     X_x = -solution_x.t
#     Sigma_x = solution_x.y[1]
#     return X_psi, Sigma_psi, X_x, Sigma_x
#
#
# def compute_shape_full(d_, Vd, config: Config) -> float:
#     X_psi, Sigma_psi, X_x, Sigma_x = shape_full(d_, config)
#     X = np.concatenate([X_psi, X_x])
#     Sigma = np.concatenate([Sigma_psi, Sigma_x])
#     return target_fun(X, Sigma, Vd)


# def pendant_drop(props: dict):
#     config = Config(props)
#     d = props.get("d")
#     if d is None:
#         res = scpo.minimize(
#             compute_shape_full,
#             x0=[d],
#             args=[config],
#             method='L-BFGS-B', jac=None,
#             options={'gtol': 1e-5, 'disp': True}
#         )
#         d = res.x[0]
#     print(f"d={d} config={config}")
#     X_psi, Sigma_psi, X_x, Sigma_x = shape_full([d], config)
#     width = np.abs(X_x.min())
#     height = np.abs(max(Sigma_psi.max(), Sigma_x.max()))
#     picScale = 16
#     plt.figure(figsize=(picScale, (height / width) * picScale))
#     plt.plot(X_psi, Sigma_psi, ".-")
#     plt.plot(X_x, Sigma_x, ".-")
#     plt.axis("equal")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.grid(True)
#     plt.savefig("Sigma_X.png", bbox_inches="tight")
#     plt.cla()
#
#
# def find_closest_point(array: ndarray, target: float) -> float:
#     # Calculate the absolute differences
#     differences = np.abs(array - target)
#     # Find the index of the minimum difference
#     closest_index = np.argmin(differences)
#     # Return the closest point
#     return array[closest_index]
#
#
# def compute_weights(x, x1, x2):
#     """
#     Compute weights based on x-coordinates.
#     Weights exponentially decay to 0 outside [x1, x2] and are close to 1 inside [x1, x2].
#     """
#     weights = np.ones_like(x)
#     # Decay factor (adjust this to control the rate of exponential decay)
#     decay_rate = 10.0
#     weights[x < x1] = np.exp(-decay_rate * (x1 - x[x < x1]))
#     weights[x > x2] = np.exp(-decay_rate * (x[x > x2] - x2))
#     return weights
#
#


