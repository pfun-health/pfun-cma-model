#!/usr/bin/env python
"""app.engine.cma_sleepwake: define the Cortisol-Melatonin-Adiponectin model.
"""
import importlib
import copy
import logging
import sys
from dataclasses import KW_ONLY, InitVar, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import (
    Any, Callable, AnyStr, Container, Dict, Iterable, Optional, Tuple
)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, validator
import pandas as pd
from scipy.optimize import Bounds, curve_fit

#: pfun imports (relative)
root_path = str(Path(__file__).parents[1])
mod_path = str(Path(__file__).parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if mod_path not in sys.path:
    sys.path.insert(0, mod_path)
try:
    from chalicelib.decorators import check_is_numpy
    from chalicelib.engine.calc import normalize
    from chalicelib.engine.data_utils import dt_to_decimal_hours
except ModuleNotFoundError:
    # TODO: Fix these imports.
    check_is_numpy = importlib.import_module(
        ".decorators", package="chalicelib").check_is_numpy
    normalize = importlib.import_module(
        ".calc", package="").normalize
    dt_to_decimal_hours = importlib.import_module(
        ".data_utils", package="").dt_to_decimal_hours

logger = logging.getLogger()


@check_is_numpy
def normalize_glucose(G, g0=70, g1=180, g_s=90):
    """Normalize glucose (mg/dL -> [0.0, 2.0]).

        <0.9: low,
        0.9: normal-low,
        0.9-1.5: normal,
        >1.5: high

    see the graph: https://www.desmos.com/calculator/ii4qrawgjo
    """
    def E(x):
        return 1.0 / (1.0 + np.exp(-2*x))
    numer = (8.95 * np.power((G - g_s), 3) +
             np.power((G - g0), 2) - np.power((G - g1), 2))
    return 2.0 * E(1e-4 * numer / (g1 - g0))


def l(x):
    return 2.0 / (1.0 + np.exp(2.0 * np.power(x, 2)))


def E(x):
    return 1.0 / (1.0 + np.exp(-2.0 * x))


def meal_distr(Cm, t, toff):
    """Meal distribution function.

    Parameters
    ----------
    Cm : float
        Cortisol concentration (mg/dL).
    t : array_like
        Time (hours).
    toff : float
        Time offset (hours).

    Returns
    -------
    array_like
        Meal distribution function.
    """
    return np.power(np.cos(2 * np.pi * Cm * (t + toff) / 24), 2)


@check_is_numpy
def K(x):
    """
    Defines the glucose response function.
    Apply a piecewise function to the input array `x`.

    Parameters:
        x (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: The result of applying the piecewise function to `x`.
    """
    return np.piecewise(x, [x > 0.0, x <= 0.0], [
        lambda x_: np.exp(-np.power(np.log(2.0 * x_), 2)), 0.0])


def vectorized_G(t, I_E, tm, taug, B, Cm, toff):
    """Vectorized version of G(t, I_E, tm, taug, B, Cm, toff).

    Parameters
    ----------
    t : array_like
        Time (hours).
    I_E : float
        Extracellular insulin (uS/mL).
    tm : array_like
        Meal times (hours).
    taug : array_like
        Meal duration (hours).
    B : float
        Bias constant.
    Cm : float
        Cortisol concentration (mg/dL).
    toff : float
        Time offset (hours).

    Returns
    -------
    array_like
        G(t, I_E, tm, taug, B, Cm, toff).
    """
    def Gtmp(tm_, taug_):
        k_G = K((t - tm_) / np.power(taug_, 2))
        return 1.3 * k_G / (1.0 + I_E)
    m = len(tm)
    n = len(t)
    j = 0
    out = np.zeros((m, n), dtype=float)
    while j < m:
        out[j, :] = Gtmp(tm[j], taug[j])
        j = j + 1
    out = out + B * (1.0 + meal_distr(Cm, t, toff))  # ! apply bias constant.
    return out


class CMASleepWakeModel:

    """Defines the Cortisol-Melatonin-Adiponectin Sleep-Wake pfun model.

    Methods:
    -------
    1) Input SG -> Project SG to 24-hour phase plane.
    2) Estimate photoperiod (t_m0 - 1, t_m2 + 3) -> Model params (d, taup).
    3) (Fit to projected SG) Compute approximate chronometabolic dynamics:
        F(m, c, a)(t, d, taup) -> ...
         ...  (+/- postprandial insulin, glucose){Late, Early}.
    """

    param_keys = ('d', 'taup', 'taug', 'B', 'Cm', 'toff')
    param_defaults = (0.0, 1.0, 1.0, 0.05, 0.0, 0.0)
    bounds = Bounds(
        lb=[-6.0, 0.5, 0.1, 0.0, 0.0, -3.0, ],
        ub=[6.0, 2.0, 3.0, 1.0, 2.0, 3.0, ],
        keep_feasible=True
    )

    def __init__(self, t=None, N=288, d=0.0, taup=1.0, taug=1.0, B=0.05,
                 Cm=0.0, toff=0.0, tM=(7.0, 11.0, 17.5),
                 seed: None | int = None, eps: float = 1e-18):
        assert (t is not None or N is not None) and (t is None or N is None), \
            "Must provide either the 't' or 'N' argument (not both)"
        if t is None:
            t = np.linspace(0, 24, num=N)
        self.t = t  # time vector
        self.tM = np.asarray(tM, dtype=float)  # mealtimes vector
        self.params = {}
        for pkey in self.param_keys:
            self.params[pkey] = locals().get(pkey)
        self.bounds = copy.copy(self.__class__.bounds)
        self.eps = eps
        self.rng = None
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

    @property
    def N(self):
        return len(self.t)

    def update(self, *args, inplace=True, **kwds):
        if len(args) > 0:
            opts = args[0]
            opts.update(kwds)
            kwds = dict(opts)
        if inplace is False:
            new_inst = copy.copy(self)
            new_inst.update(inplace=True, **kwds)
            return new_inst
        #: ! handle case in which taug was given as a vector initially
        if 'taug' in kwds and isinstance(self.params['taug'], Container):
            taug_new = kwds.pop('taug')
            match isinstance(taug_new, Container):
                case True:
                    #: ! replace current values elementwise if given a vector
                    self.params['taug'] = np.broadcast_to(
                        taug_new, (self.n_meals, ))
                case False:  # ! else, taug is a scale: <old_taug> *= new_taug
                    self.params['taug'] = np.array(
                        self.params['taug'], dtype=float) * float(taug_new)
        #: update all given params
        self.params.update({k: kwds[k] for k in kwds if k in self.params})
        self.params = {k: self.params[k]
                       for k in self.param_keys}  # ! ensure order
        if 'tM' in kwds:
            self.tM = np.array(kwds['tM'], dtype=float).flatten()
        if 'N' in kwds:
            self.t = np.linspace(0, 24, num=kwds['N'])
        if 'seed' in kwds:
            self.rng = np.random.default_rng(seed=kwds['seed'])
        if 'eps' in kwds:
            self.eps = kwds['eps']

    @property
    def d(self) -> float:
        return self.params.get("d")

    @property
    def taup(self):
        return self.params.get("taup")

    @property
    def n_meals(self):
        return len(self.tM)

    @property
    def taug(self) -> np.ndarray:
        """taug: get an array broadcasted to: (, number_of_meals)."""
        taug_ = self.params.get("taug")
        taug_vector = np.broadcast_to(taug_, (self.n_meals, ))
        return taug_vector

    @property
    def B(self) -> float:
        """Return the current bias parameter value (B)."""
        return self.params.get("B")

    @property
    def Cm(self) -> float:
        """return the current Cm param value."""
        return self.params.get("Cm")

    @property
    def toff(self) -> float:
        return self.params.get("toff")

    def E_L(self, t=None):
        if t is None:
            t = self.t
        return l(0.025 * np.power((t - 12.0 - self.d), 2) /
                 (self.eps + self.taup))

    @property
    def L(self):
        return self.E_L(t=self.t)

    def M(self, t=None):
        """compute the estimated relative Melatonin signal."""
        if t is None:
            t = self.t
        m_out = np.power((1.0 - self.L), 3) * \
            np.power(np.cos(-(t - 3.0 - self.d) * np.pi / 24.0), 2)
        if self.rng is not None:
            # ! tiny amount of random noise
            m_out = m_out + \
                self.rng.uniform(low=-self.eps, high=self.eps, size=self.N)
        return m_out

    @property
    def m(self):
        return self.M(t=self.t)

    @property
    def c(self):
        return (4.9 / (1.0 + self.taup)) * np.pi * E(np.power((self.L - 0.88), 3)) * \
            E(0.05 * (8.0 - self.t + self.d)) * E(2.0 * np.power(-self.m, 3))

    @property
    def a(self):
        return (E(np.power((-self.c * self.m), 3)) +
                np.exp(-0.025 * np.power((self.t - 13 - self.d), 2)) *
                self.E_L(t=0.7*(27-self.t+self.d))) / 2.0

    @property
    def I_S(self):
        return 1.0 - 0.23 * self.c - 0.97 * self.m

    @property
    def I_E(self):
        return self.a * self.I_S

    @property
    def G(self):
        return vectorized_G(self.t, self.I_E, self.tM, self.taug, self.B,
                            self.Cm, self.toff)

    @property
    def g(self):
        """g: get the per-meal post-prandial glucose dynamics.

        Examples:
        ---------
            >>> cma = CMASleepWakeModel(N=10, taug=[1.0, 2.0, 3.0])
            >>> cma.g
            array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.73725968e-01,
                    1.57882217e-02, 1.27544618e-03, 2.05703931e-04, 5.09901523e-05,
                    1.50385708e-05, 4.96125002e-06],
                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                    0.00000000e+00, 7.85187035e-01, 3.77414036e-01, 1.70718635e-01,
                    7.87996469e-02, 3.75426497e-02],
                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.67894822e-01,
                    1.13923320e+00, 1.08996350e+00]])
        """
        return self.G

    def integrate_signal(self, signal: np.ndarray | None = None,
                         signal_name: str | None = None,
                         t0: int | float = 0, t1: int | float = 24,
                         M: int = 3,
                         t_extra: Tuple | None = None,
                         tvec: np.ndarray | None = None):
        """Integrate the signal between the hours given, assuming M discrete events.

            t_extra specifies any additional range of 'accepted hours' as an inclusive tuple [te0, te1],
            to be included in the target time period.
        """
        assert any([(signal is None), (signal_name is None)]
                   ), "Must provide exactly one of signal or signal_name"
        assert any([(signal is not None), (signal_name is not None)]
                   ), "Must provide exactly one of signal or signal_name"
        if tvec is None:
            tvec = self.t
        if signal_name is not None:
            signal = getattr(self, signal_name)
        if signal.shape[0] != tvec.size:
            signal = signal.T
        period = np.logical_and((tvec >= t0), (tvec <= t1))
        if t_extra is not None:
            period = np.logical_or(
                period, (tvec >= t_extra[0]) & (tvec <= t_extra[1]))
        total = np.nansum(signal[period]) / (M * (t1 - t0))
        return total

    def morning(self, signal: np.ndarray = None, signal_name=None):
        """compute the total morning integrated signal."""
        return self.integrate_signal(signal=signal, signal_name=signal_name, t0=4, t1=13)

    def evening(self, signal: np.ndarray = None, signal_name=None):
        """Compute the total evening integrated signal."""
        return self.integrate_signal(signal=signal, signal_name=signal_name,
                                     t0=16, t1=24, t_extra=(0, 3))

    @property
    def columns(self):
        return ["t", "c", "m", "a", "I_S", "I_E", "L", "G"]

    @property
    def g_morning(self):
        return self.morning(self.g)

    @property
    def g_evening(self):
        return self.evening(self.g)

    @property
    def g_instant(self):
        """vector of instantaneous (overall) glucose."""
        return np.nansum(self.g, axis=0)

    @property
    def df(self) -> pd.DataFrame:
        return self.run()

    @property
    def dt(self):
        #: TimedeltaIndex (in hours)
        return pd.to_timedelta(self.t, unit='H')

    @classmethod
    def get_model_args(cls):
        """for maintaining compatibility with the pfun.model_funcs API"""
        return dict(zip(cls.param_keys, cls.param_defaults))

    @classmethod
    def fit(cls, *args, **kwds):
        return fit_model(*args, **kwds)

    def run(self) -> pd.DataFrame:
        """run the model, return the solution as a labeled pd.DataFrame.

        Examples:
        ---------
            >>> cma = CMASleepWakeModel(N=4)
            >>> df = cma.run()
            >>> print(tabulate.tabulate(df, floatfmt='.3f', headers=df.columns))
                     t      c      m      a    I_S    I_E    g_0    g_1    g_2      G
            --  ------  -----  -----  -----  -----  -----  -----  -----  -----  -----
             0   0.000  0.083  0.854  0.251  0.153  0.038  0.000  0.000  0.000  0.000
             1   8.000  0.962  0.003  0.517  0.776  0.401  0.574  0.000  0.000  0.574
             2  16.000  0.597  0.000  0.565  0.863  0.488  0.000  0.004  0.000  0.005
             3  24.000  0.020  0.854  0.250  0.167  0.042  0.000  0.000  0.002  0.002
        """
        #: init list of "standard" columns
        columns = list(self.columns)
        # ! exclude instantaneous G until after computing components...
        columns.remove("G")
        #: get the corresponding values
        values = [getattr(self, k) for k in columns]
        #: compute "G" (separate components)
        g = self.g
        #: labels & values of the separate components of "G"
        gi_cols = [f"g_{j}" for j in range(g.shape[0])]
        columns = columns + gi_cols
        values = values + [g[i, :] for i in range(g.shape[0])]
        data = {k: v for k, v in zip(columns, values)}
        df = pd.DataFrame(data, columns=columns, index=self.dt)
        #: record instantaneous glucose
        df["G"] = self.g_instant
        #: record estimated meal times
        ismeal = [(df['t'] - tm).abs().idxmin() for tm in self.tM]
        df['is_meal'] = False
        df.loc[ismeal, 'is_meal'] = True
        return df


def round_to_nearest_integer(number):
    rounded_number = round(number, 2)
    return int(rounded_number)


class CMAUtils:

    @staticmethod
    def get_hour_of_day(
            hour: Tuple[float | int] | float | int) -> int | str | Tuple[str | int]:
        """
        Get the hour of the day based on the given hour value.

        Parameters:
            hour (float or int): The hour value to convert.

        Returns:

            int: The hour of the day as an integer.

            OR

            str: The hour of the day in the format '12AM', '12PM', '1AM', '1PM', etc.

            ...OR as a tuple.

        Raises:
            ValueError: If the hour is not a float or integer value, or if it is not between 0 and 24.
        """
        if isinstance(hour, tuple):
            # ! handle tuple
            return tuple(map(CMAUtils.get_hour_of_day, hour))
        if not isinstance(hour, (float, int)):
            raise ValueError("The hour must be a float or integer value.")
        if hour < 0 or hour > 24:
            raise ValueError("The hour must be between 0 and 24.")
        if hour == 0 or hour == 24:
            return '12AM'
        elif hour == 12:
            return '12PM'
        elif hour < 12:
            return f'{int(hour)}AM'
        else:
            return f'{int(hour) - 12}PM'

    @staticmethod
    def label_meals(df: pd.DataFrame,
                    rounded: [None | Callable] = round_to_nearest_integer,
                    as_str: bool = False) -> Tuple[str | int | float]:
        """Label the meal times in a CMA model results dataframe.
        Parameters:
            df (pd.DataFrame): The CMA model results dataframe.
            rounded (None or Callable): Function to round the meal times.
            as_str (bool): If True, return the meal times as strings.
        Returns:
            tuple: The meal times as strings or integers.
        Examples:
            >>> df = pd.DataFrame({'t': [0, 8, 16, 24], 'is_meal': [True, True, True, False]})
            >>> CMAUtils.label_meals(df)
            ('12AM', '8AM', '4PM')
            >>> CMAUtils.label_meals(df, as_str=True)
            ('12AM', '8AM', '4PM')
            >>> CMAUtils.label_meals(df, rounded=round_to_nearest_integer)
            (0, 8, 16)
        """
        #: get the meal times
        mealtimes = df.loc[df['is_meal'], 't']
        if rounded is not None:
            mealtimes = mealtimes.apply(rounded)
        tM = tuple(mealtimes)
        if as_str:
            tM = CMAUtils.get_hour_of_day(tM)
        return tuple(tM)


@dataclass
class CMALabeledTimeAxis:

    """Labeled time axis for CMA model results.

    Description: Label strings, including hour of day, meal times...
    ...in a CMA model results dataframe.
    """

    df: InitVar[pd.DataFrame | None] = None
    KW_ONLY
    t: Optional[Tuple[int | float]] = ...
    tM: InitVar[Optional[np.ndarray | Tuple[int | float]]] = ...
    t_labels: Optional[Tuple[str]] = ...
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self, df: pd.DataFrame | None, t=None, tM=None, t_labels=None,
                      extra_kwargs={}) -> None:
        if t is None and df is None:
            raise ValueError("Either t or df must be specified.")
        self.t = t
        self.tM = tM
        self.t_labels = t_labels
        if self.t is None:
            self.t = df.index.to_numpy()
        if self.tM is None and df is not None:
            self.tM = CMAUtils.label_meals(df, as_str=False)
        if self.t_labels is None:
            self.t_labels = CMAUtils.get_hour_of_day(self.t)

    @property
    def xaxis_tlabels(self):
        return (self.t, self.t_labels)

    @property
    def xaxis_tMlabels(self):
        return (self.tM, CMAUtils.label_meals(self.df, as_str=True))


@dataclass
class CMAPlotConfig:
    """configuration for plotting the CMA model results"""

    df: pd.DataFrame | None = None
    plot_cols: Optional[Tuple[str]] = (
        "g_0", "g_1", "g_2", "G", "c", "m", "a", "L", "I_S", "I_E",
        "is_meal", "g_raw")
    labels: Optional[Tuple[str]] = ("Breakfast", "Lunch", "Dinner",
                                    "Glucose", "Cortisol", "Melatonin",
                                    "Adiponectin", "Photoperiod (irradiance)",
                                    "Insulin (secreted)",
                                    "Insulin (effective)",
                                    "Meals",
                                    "Glucose (Data)")
    colors: Optional[Tuple[str]] = (
        "#ec5ef9",
        "#bd4bc7",
        "#8b3793",
        "purple",
        "cyan",
        "darkgrey",
        "m",
        'tab:orange',
        'tab:red',
        'red',
        'k',
        'darkgrey'
    )

    @classmethod
    def get_label(cls, col: Container | AnyStr):
        if not isinstance(col, str):
            return [cls.get_label(c) for c in col]
        index = cls.plot_cols.index(col)
        return cls.labels[index]

    @classmethod
    def get_color(cls, col: Container | AnyStr, rgba=False, as_hex=False,
                  keep_alpha=False):
        if not isinstance(col, str):
            return [cls.get_color(c, rgba=rgba) for c in col]
        try:
            index = cls.plot_cols.index(col)
            c = cls.colors[index]
        except (IndexError, ValueError) as excep:
            msg = f"failed to find a plot color for: {col}"
            logging.warn(msg, exc_info=1)
            raise excep.__class__(msg)
        if rgba is True or as_hex is True:
            c = matplotlib.colors.to_rgba(c)
            if as_hex is True:
                c = matplotlib.colors.rgb2hex(c, keep_alpha=keep_alpha)
        return c

    @classmethod
    def set_global_axis_properties(cls, axs, df=None):
        """set universal axis properties (like time of day labels for x-axis).
        """
        if df is None:
            df = cls.df
        for ax in axs:
            ax.set_xticks(*CMALabeledTimeAxis(df).xaxis_tlabels)
            ax.set_xlim([0.01, 23.99])
            ax.set_xlabel("Time (24-hours)")
        return axs

    @classmethod
    def set_global_axis_attributes(cls, axs):
        """alias for set_global_axis_properties..."""
        return cls.set_global_axis_properties(axs)

    @classmethod
    def plot_model_results(cls, df=None, soln=None, plot_cols=None,
                           separate2subplots=False, as_blob=True):
        """plot the results of the model"""
        if df is None:
            df = cls.df
        if soln is None:
            soln = cls.soln
        if plot_cols is None:
            plot_cols = cls.plot_cols
        #: drop is_meal from plot cols... (it's bool afterall)
        plot_cols = list(plot_cols)
        if "is_meal" in plot_cols:
            ismeal_ix = plot_cols.index("is_meal")
            plot_cols.pop(ismeal_ix)
        #: combine the data into a single dataframe
        df = df.set_index("t")
        soln = soln.set_index("t")
        df = pd.merge_ordered(df.copy(), soln, suffixes=("", "_soln"), on="t")
        df = df.set_index("t")
        df = df.drop(columns=['time', ]).interpolate(method='akima')
        fig, axs = plt.subplots(
            nrows=2 if separate2subplots is False else len(plot_cols) + 1)
        #: plot meal times, meal sizes
        ax = axs[0]
        ax.fill_between(df.index, y1=df['G_soln'].min(
        ), y2=df["G_soln"], color='k', label="Estimated Meal Size")
        ax.vlines(x=df.loc[df.is_meal.astype(float).fillna(0.0) > 0].index,
                  ymin=ax.get_ylim()[0], ymax=df.G_soln.max(),
                  color='r', lw=3, linestyle='--', label='estimated mealtimes')
        ax.legend()
        #: plot other traces
        if separate2subplots is False:
            df.plot.area(y=plot_cols, color=cls.get_color(plot_cols), ax=axs[1],
                         alpha=0.2, label=cls.get_label(plot_cols), stacked=True)
        elif separate2subplots is True:
            for pcol, axi in zip(plot_cols, axs[1:]):
                axi.fill_between(
                    x=df.index, y1=df[pcol].min(), y2=df[pcol],
                    color=cls.get_color(pcol),
                    alpha=0.2,
                    label=cls.get_label(pcol)
                )
                axi.legend()
        #: set global properties for all axes...
        axs = cls.set_global_axis_properties(axs)
        fig.tight_layout()
        #: return the figure and axes (unless this is to be a blob for the web)
        if as_blob is False:
            return fig, axs
        bio = BytesIO()
        fig.savefig(bio, format='png')
        bio.seek(0)
        bytes_value = bio.getvalue()
        img_src = 'data:image/png;base64,'
        img_src = img_src + b64encode(bytes_value).decode('utf-8')
        plt.tight_layout()
        plt.close()
        return img_src


class CMAFitResult(BaseModel):
    soln: pd.DataFrame
    cma: CMASleepWakeModel
    popt: np.ndarray
    pcov: np.ndarray
    infodict: Dict
    mesg: str
    ier: int
    popt_named: Optional[Dict]

    @validator("popt_named", always=True, allow_reuse=True)
    def _get_popt_named(cls, v, values):
        cma = values.get("cma")
        popt = values.get("popt")
        return {k: v for k, v in zip(cma.param_keys, popt)}

    class Config:
        arbitrary_types_allowed = True


def estimate_mealtimes(data, ycol: str = 'G', tm_freq: str = "2h", n_meals: int = 4, **kwds):
    n_meals = int(n_meals)
    df = data[['t', ycol]]
    if not isinstance(df.index, pd.TimedeltaIndex):
        df = df.assign(dt=pd.to_timedelta(df["t"], "H"))
        df.set_index("dt", inplace=True)
    dfres = df.resample(tm_freq).mean()
    tM = dfres[ycol].diff().dropna() \
        .groupby(pd.Grouper(freq=tm_freq)).max() \
        .sort_values() \
        .index.to_series().apply(
        lambda d: dt_to_decimal_hours(d)
    ).unique()[-n_meals:] - 0.05
    tM[tM < 0.0] += 23.9999
    tM[tM > 24.0] -= 23.9999
    tM.sort()
    return tM


def fit_model(data: pd.DataFrame, ycol: str = "G", tM: None | Iterable = None,
              tm_freq: str = "2h", curve_fit_kwds: Dict = {}, **kwds) -> CMAFitResult:
    """use `scipy.optimize.curve_fit` to fit the model to data

    Arguments:
    ----------
    - data (pd.DataFrame) : ["t", "ycol"]
        "t"      : 24-hour hour of day
        "<ycol>" : raw egv
    - ycol (str) : name of output data column
    - tM (optional) : vector of mealtimes (decimal hours).
        If unspecified, mealtimes will be estimated (default).
    """

    #: update from keywords
    default_cf_kwds = {
        "verbose": 0,
        "ftol": 1e-6,
        "xtol": 1e-6,
        "max_nfev": data[ycol].size * 500,
        "check_finite": True,
        "absolute_sigma": False,
        "x_scale": [2.0, 0.01, 0.01, 0.1, 0.1, 0.1],
    }
    default_cf_kwds.update(curve_fit_kwds)
    curve_fit_kwds = dict(default_cf_kwds)

    #: get xdata, ydata as float vectors
    xdata = data["t"].to_numpy(dtype=float)
    ydata = data[ycol].to_numpy(dtype=float, na_value=np.nan)

    #: estimate tM if needed
    if tM is None:
        tM = estimate_mealtimes(data, ycol, **kwds)

    #: instantiate model
    cma = CMASleepWakeModel(t=xdata, N=None, tM=tM, **kwds)
    if curve_fit_kwds.get("verbose"):
        print("taup0=", cma.taup)

    def fun(xdata, d, taup, taug, B, Cm, toff, cma=cma):
        cma.update(inplace=True, d=d, taup=taup,
                   taug=taug, B=B, Cm=Cm, toff=toff)
        return cma.g_instant

    #: perform fitting
    pkeys_include = cma.param_keys
    p0 = [cma.params[k] for k in pkeys_include]
    if isinstance(p0[cma.param_keys.index("taug")], Container):
        #: ! ensure we update using a scalar
        p0[cma.param_keys.index("taug")] = 1.0
    bounds = cma.bounds
    popt, pcov, infodict, mesg, ier = curve_fit(
        fun, xdata, ydata, p0=p0, bounds=bounds, full_output=True,
        **curve_fit_kwds)

    #: informed model (best fit)
    p0_cma = dict(zip(pkeys_include, popt, strict=True))
    cma = cma.update(inplace=False, **p0_cma)

    return CMAFitResult(soln=cma.df, cma=cma, popt=popt, pcov=pcov,
                        infodict=infodict, mesg=mesg, ier=ier)


def test_fit_model(n=288, plot=False, opts=None, **kwds):
    N = n
    curve_fit_kwds = opts or {}
    if not isinstance(curve_fit_kwds, dict):
        curve_fit_kwds = dict(curve_fit_kwds)
    curve_fit_kwds.update({"verbose": 1})
    print("\nrunning test_fit_model...")
    #: creating the "fake data" initial simulation
    cma = CMASleepWakeModel(
        N=N, **{k: kwds[k] for k in kwds if k in CMASleepWakeModel.param_keys})
    cma.update(**kwds)
    df = cma.run()
    sunvals = df.L.loc[np.isclose(
        df.L.values, 0.5, atol=0.1)].index.components.hours
    sunrise, sunset = sunvals.iat[0], sunvals.iat[-1]
    taup_est = (sunset - sunrise) / 12.0
    print()
    print("sunrise_act=", sunrise, "sunset_act=", sunset)
    print("tM_actual=\n", cma.tM)
    print("p_actual=\n", cma.params)
    print()
    fit_result = fit_model(
        df, ycol="G", tm_freq="15T", curve_fit_kwds=curve_fit_kwds,
        taup=taup_est)
    print()
    print("tM_est=\n", fit_result.cma.tM)
    print("p_opt=\n", fit_result.cma.params)
    print()
    print("...done running test_fit_model.\n")
    if plot is True:
        import matplotlib.pyplot as plt
        plt.ion()
        ax = df.plot(x="t", y=["L", "G"], label=["L", "G"])
        ax = fit_result.soln.plot(
            x="t", y=["L", "G"], ax=ax, label=["L_opt", "G_opt"])
        ax.vlines(x=fit_result.soln.loc[fit_result.soln.is_meal, 't'],
                  ymin=0, ymax=1, color='k', lw=3, linestyle='--', label='meal')
        ax.legend()
        ax.set_title(f"n={n:02d}")
        ax.set_xticklabels(ax.get_xticks(), rotation=45)
        ax.get_figure().savefig("./test_fit_chalicelib.png")
    return fit_result


def process_kwds(ctx, param, value):
    if param.name != "opts":
        return value
    value = list(value)
    for i in range(len(value)):
        value[i] = list(value[i])
        if value[i][1].isnumeric():
            try:
                new = int(value[i][1])
            except ValueError:
                new = float(value[i][1])
            finally:
                value[i][1] = new
    return value
