from logging import Logger

import numpy as np
import scipy.integrate as scp_integrate
import scipy.interpolate as scp_interpolate
import scipy.optimize as scpo
from matplotlib import pyplot as plt
from numpy import ndarray
from sciform import Formatter
sform = Formatter(
    round_mode="sig_fig",  # This is the default
    ndigits=4,
)

# s1 = -1 for pendant drop
# s2 = 1 for rho_d - rho_a > 0
iter = 0

class Config:
    def __init__(self, props: dict, logging: Logger):
        self.s1: int = props.get("s1", -1)
        self.s2: int = props.get("s2", 1)
        self.curvature = props.get("curvature", 0)
        self.B: float = 2*self.curvature
        self.d: float = props.get("d", 1)
        self.psi1: float = 0
        self.psi2: float = props.get("psi2", np.pi/2)
        second_tip = props.get("second_tip", {})
        coords_left = np.c_[second_tip["xx_left"], second_tip["yy_left"]]
        self.coords_left = coords_left
        coords_right = np.c_[second_tip["xx_right"], second_tip["yy_right"]]
        self.coords_right = coords_right
        self.coords = None
        self.coord_sign = 1
        self.x0 = second_tip["x0"]
        self.y0 = second_tip["y0"]
        self.logging = logging
        self.gradient_pressure_coef: float = props.get("sigma")/(props.get("diam")*props.get("rho1")*props.get("Umean")**2)
        self.gradient_pressure_coef_dim: float = props.get("sigma")/props.get("diam")**2
        self.length_left = None
        self.length_right = None
        self.length_avg = None
        logging.info(f"B={self.B}, x0={self.x0}, y0={self.y0}")

    def set_coord(self, coords):
        self.coords = coords
        self.coord_sign = 1 if (coords[0, 1] + coords[1, 1] > 0) else -1

    def get_pressure_gradient(self, length: float):
        return [
            sform(self.gradient_pressure_coef/length**2),
            sform(self.gradient_pressure_coef_dim/length**2)
        ]

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

def fun_x(x: float, z: tuple[float, float], length: float, config: Config) -> list[float, float]:
    w, f = z
    return [
        (1 + w**2) * (1 / f + np.sqrt(1 + w**2) * (config.s1 * config.s2 * x / length**2 - config.B)),
        w
    ]


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
    Sigma_psi = solution_psi.y[1]*config.coord_sign
    return X_psi, Sigma_psi


def full_shape_psi(length: float, config: Config) -> tuple[ndarray, ndarray]:
    config.set_coord(config.coords_left)
    X_psi_left, Sigma_psi_left = shape_psi(length, config)
    config.set_coord(config.coords_right)
    X_psi_right, Sigma_psi_right = shape_psi(length, config)
    if np.abs(X_psi_left[-1] - X_psi_right[0]) < 0.01:
        X_psi = np.concatenate([X_psi_left[::-1], X_psi_right])
        Sigma_psi = np.concatenate([Sigma_psi_left[::-1], Sigma_psi_right])
    else:
        X_psi = np.concatenate([X_psi_right[::-1], X_psi_left])
        Sigma_psi = np.concatenate([Sigma_psi_right[::-1], Sigma_psi_left])
    return X_psi, Sigma_psi



# def compute_shape_psi(Vd, config: Config) -> float:
#     X_psi, Sigma_psi = shape_psi(config)
#     return target_fun(X_psi, Sigma_psi, Vd)


def find_difference(x: ndarray, y: ndarray, xn: ndarray, yn: ndarray) -> float:
    x_min = x.min()
    xn_min = xn.min()
    if x_min < xn_min:
        ind1 = x < xn_min
        int0 = np.abs(scp_integrate.simps(y[ind1], x=x[ind1], axis=-1, even='avg'))
        ind2 = x >= xn_min
        int1 = np.abs(scp_integrate.simps(y[ind2], x=x[ind2], axis=-1, even='avg'))
        int2 = np.abs(scp_integrate.simps(yn, x=xn, axis=-1, even='avg'))
        return (int0 + np.abs(int1 - int2))**2
    else:
        ind1 = xn < x_min
        int0 = np.abs(scp_integrate.simps(yn[ind1], x=xn[ind1], axis=-1, even='avg'))
        ind2 = xn >= x_min
        int1 = np.abs(scp_integrate.simps(yn[ind2], x=xn[ind2], axis=-1, even='avg'))
        int2 = np.abs(scp_integrate.simps(y, x=x, axis=-1, even='avg'))
        return (int0 + np.abs(int1 - int2))**2


def plot_between(x, y, xn, yn, file_name: str):
    global iter
    x1 = x.copy()
    y1 = y.copy()
    x2 = xn.copy()
    y2 = yn.copy()
    x1_min = x1.min()
    x2_min = x2.min()
    if x1_min < x2_min:
        x2 = np.insert(x2, 0, [x1_min, x2_min-0.0001])
        y2 = np.insert(y2, 0, [0, 0])
    else:
        x1 = np.insert(x1, 0, [x2_min, x1_min-0.0001])
        y1 = np.insert(y1, 0, [0, 0])

    plt.figure(figsize=(2/3, 2/3))
    # Combine x-coordinates
    x_union = np.union1d(x1, x2)

    # Interpolate y-values
    interp_y1 = scp_interpolate.interp1d(x1, y1, kind='linear', fill_value='extrapolate')
    interp_y2 = scp_interpolate.interp1d(x2, y2, kind='linear', fill_value='extrapolate')

    y1_interp = interp_y1(x_union)
    y2_interp = interp_y2(x_union)

    # Plot and fill between
    plt.plot(x1, y1, 'b-', label='Curve 1')
    plt.plot(x2, y2, 'r-', label='Curve 2')

    plt.fill_between(x_union, y1_interp, y2_interp, color='gray', alpha=0.5, label='Filled Area')
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.axis('equal')
    # plt.savefig(file_name, bbox_inches="tight", pad_inches=0, dpi=1200, transparent=False)
    plt.savefig(file_name[:-3] + "pdf", bbox_inches="tight", pad_inches=0, transparent=False)
    plt.close()
    iter += 1


def fit_curve_err(par, config: Config, part: str, mode: str):
    if mode == "length":
        length, = par
    else:
        length, B = par
        config.B = B
    if part == "left":
        config.set_coord(config.coords_left)
    elif part == "right":
        config.set_coord(config.coords_right)
    elif part == "both":
        err = fit_curve_err(par, config, part="right", mode=mode) + fit_curve_err(par, config, part="left", mode=mode)
        config.logging.info({"fit_curve_err": par, "part": part, "mode": mode, "err": err})
        return err
    else:
        raise NotImplemented
    X_psi, Sigma_psi = shape_psi(length, config)

    err = find_difference(
        X_psi,
        Sigma_psi,
        config.coords[:, 0],
        config.coords[:, 1],
    )
    config.logging.info({"fit_curve_err": par, "part": part, "mode": mode, "err": err})
    return err


def fit_curve_part(x0: tuple[float], config: Config, mode: str, part: str):
    res = scpo.minimize(
        fit_curve_err,
        x0=x0,
        args=(config, part, mode),
        method="Nelder-Mead",
        # method="L-BFGS-B",
        # options={"gtol": 1e-10, "disp": True, "maxiter": 2000, "eps": 1e-10},
    )
    if mode == "length":
        length, = res.x
    else:
        length, B = res.x
        config.B = B
    if part == "left":
        config.length_left = length
    elif part == "right":
        config.length_right = length
    else:
        config.length_avg = length
    config.logging.info({f"length_{part}": length, "B": config.B, "mode": mode, "part": part})

    return shape_psi(length, config)


# unknown variable is capillary length `length`
def fit_curve(x0: tuple[float], config: Config, mode: str):
    X_psi_left, Sigma_psi_left = fit_curve_part(x0, config, mode=mode, part="left")
    X_psi_right, Sigma_psi_right = fit_curve_part(x0, config, mode=mode, part="right")
    X_psi_both, Sigma_psi_both = fit_curve_part(x0, config, mode=mode, part="both")

    # plt.plot(X_psi_left, Sigma_psi_left, "-", color='red')
    # plt.plot(X_psi_right, Sigma_psi_right, "-", color='blue')
    # plt.grid(True)
    # plt.savefig("shape_full.eps", bbox_inches="tight")
    # plt.close()
    if np.abs(X_psi_left[-1] - X_psi_right[0]) < 0.01:
        X_psi = np.concatenate([X_psi_left[::-1], X_psi_right])
        Sigma_psi = np.concatenate([Sigma_psi_left[::-1], Sigma_psi_right])
    else:
        X_psi = np.concatenate([X_psi_right[::-1], X_psi_left])
        Sigma_psi = np.concatenate([Sigma_psi_right[::-1], Sigma_psi_left])
    return X_psi, Sigma_psi, X_psi_both, Sigma_psi_both


def shape_full(length: float, config: Config) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    X_psi, Sigma_psi = shape_psi(length, config)
    X0 = X_psi[-1]
    W0 = (Sigma_psi[-1] - Sigma_psi[-2]) / (X_psi[-1] - X_psi[-2])
    f0 = Sigma_psi[-1]
    config.logging.info({
        "X0": X0,
        "W0": W0,
        "f0": f0,
        "length": length,
    })
    solution_x = scp_integrate.solve_ivp(
        fun=fun_x,
        t_span=(X0, config.d),
        y0=(W0, f0),
        method="BDF",
        args=(length, config),
        min_step=1e-6,
        max_step=1e-3,
        rtol=1e-15,
        dense_output=True,
    )
    X_x = solution_x.t
    Sigma_x = solution_x.y[1]
    return X_psi, Sigma_psi, X_x, Sigma_x


def full_shape_psi_x(
        length: float,
        config: Config,
        mirror: bool = True,
        separate: bool = False
) -> tuple[ndarray, ndarray] | tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    config.set_coord(config.coords_left)
    X_psi_left, Sigma_psi_left, X_x_left, Sigma_x_left = shape_full(length, config)

    # mirror the left solution, otherwise separately compute for right
    if mirror:
        X_psi_right, Sigma_psi_right, X_x_right, Sigma_x_right = X_psi_left, -Sigma_psi_left, X_x_left, -Sigma_x_left
    else:
        config.set_coord(config.coords_right)
        X_psi_right, Sigma_psi_right, X_x_right, Sigma_x_right = shape_full(length, config)
    if separate:
        return X_psi_left, Sigma_psi_left, X_x_left, Sigma_x_left, X_psi_right, Sigma_psi_right, X_x_right, Sigma_x_right

    if np.abs(X_psi_left[-1] - X_psi_right[0]) < 0.01:
        X_psi_x = np.concatenate([X_x_left[::-1], X_psi_left[::-1], X_psi_right, X_x_right])
        Sigma_psi_x = np.concatenate([Sigma_x_left[::-1], Sigma_psi_left[::-1], Sigma_psi_right, Sigma_x_right])
    else:
        X_psi_x = np.concatenate([X_x_right[::-1], X_psi_right[::-1], X_psi_left, X_x_left])
        Sigma_psi_x = np.concatenate([Sigma_x_right[::-1], Sigma_psi_right[::-1], Sigma_psi_left, Sigma_x_left])
    return X_psi_x, Sigma_psi_x
