#!/usr/bin/env python
"""app.engine.cma_sleepwake: define the Cortisol-Melatonin-Adiponectin model.
"""
import importlib
import copy
import logging
import sys
from dataclasses import KW_ONLY, InitVar, dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, AnyStr, Container, Dict, Iterable, Optional, Tuple
)
import numpy as np

#: pfun imports (relative)
root_path = str(Path(__file__).parents[1])
mod_path = str(Path(__file__).parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if mod_path not in sys.path:
    sys.path.insert(0, mod_path)
try:
    from chalicelib.decorators import check_is_numpy
    from chalicelib.engine.calc import normalize, normalize_glucose
    from chalicelib.engine.data_utils import dt_to_decimal_hours
    from chalicelib.engine.bounds import Bounds
    import pandas as pd
except ModuleNotFoundError:
    check_is_numpy = importlib.import_module(
        ".decorators", package="chalicelib").check_is_numpy
    normalize = importlib.import_module(
        ".calc", package="chalicelib.engine").normalize
    normalize_glucose = importlib.import_module(
        ".calc", package="chalicelib.engine").normalize_glucose
    dt_to_decimal_hours = importlib.import_module(
        ".data_utils", package="chalicelib.engine").dt_to_decimal_hours
    Bounds = importlib.import_module(
        ".bounds", package="chalicelib.engine").Bounds

logger = logging.getLogger()


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
        self.params.update({k: kwds[k] for k in kwds if k in self.param_keys})
        self.params = {k: self.params[k] for k in self.param_keys}  # ! ensure order
        #: keep within specified bounds
        if self.bounds.keep_feasible is True:
            self.params = self.bounds.update_values(self.params)
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
    
    @property
    def pvec(self):
        """easy access to parameter vector (copy)"""
        return np.array([self.params[k] for k in self.param_keys])

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
