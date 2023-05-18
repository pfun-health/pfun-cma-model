#!/usr/bin/env python
"""app.engine.cma_sleepwake: define the Cortisol-Melatonin-Adiponectin model"""
import copy
import json
import logging
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import AnyStr, Container, Dict, Iterable, Optional, Tuple
import importlib

import click
import matplotlib
import numba
import numpy as np
import pandas as pd
import tabulate
from pydantic import BaseModel, validator
from scipy.optimize import Bounds, curve_fit
from scipy.signal import find_peaks
from sklearn.model_selection import ParameterGrid

logger = logging.getLogger()


#: pfun imports (relative)
top_path = Path(__file__).parents[2]
root_path = Path(__file__).parents[1]
for pth in [top_path, root_path]:
    pth = str(pth)
    if pth not in sys.path:
        sys.path.insert(0, pth)


dt_to_decimal_hours = importlib.import_module(".data_utils", "engine").dt_to_decimal_hours
normalize = importlib.import_module(".calc", "engine").normalize
check_is_numpy = importlib.import_module(".decorators", "app").check_is_numpy


@check_is_numpy
def normalize_glucose(G, g0=70, g1=180, g_s=90):
    """Normalize glucose (mg/dL -> [0.0, 2.0])

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
    return np.power(np.cos(2 * np.pi * Cm * (t + toff) / 24), 2)


@check_is_numpy
def K(x):
    return np.piecewise(x, [x > 0.0, x <= 0.0,], [lambda x_: np.exp(-np.power(np.log(2.0 * x_), 2)), 0.0])


def vectorized_G(t, I_E, tm, taug, B, Cm, toff):
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
    --------
    1) Input SG -> Project SG to 24-hour phase plane.
    2) Estimate photoperiod (t_m0 - 1, t_m2 + 3) -> Model params (d, taup).
    3) (Fit to projected SG) Compute approximate chronometabolic dynamics:
        F(m, c, a)(t, d, taup) -> (+/- postprandial insulin, glucose){Late, Early}.
    """

    param_keys = ('d', 'taup', 'taug', 'B', 'Cm', 'toff')
    param_defaults = (0.0, 1.0, 1.0, 0.05, 0.0, 0.0)
    bounds = Bounds(
        lb=[-6.0, 0.5, 0.1, 0.0, 0.0, -3.0, ],
        ub=[6.0, 2.0, 3.0, 1.0, 2.0, 3.0, ],
        keep_feasible=True
    )

    def __init__(self, t=None, N=288, d=0.0, taup=1.0, taug=1.0, B=0.05, Cm=0.0, toff=0.0,
                 tM=np.array([7.0, 11.0, 17.5]), seed: None | int = None, eps: float = 1e-18):
        assert (t is not None or N is not None) and \
            (t is None or N is None), "Must provide either the 't' or 'N' argument (not both)"
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
                    #: ! replace the current values elementwise if given a vector
                    self.params['taug'] = np.broadcast_to(
                        taug_new, (self.n_meals, ))
                case False:  # ! otherwise, treat the given taug as a scale: <old_taug> *= new_taug
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
        """return the current bias parameter value (B)."""
        return self.params.get("B")

    @property
    def Cm(self) -> float:
        """return the current Cm param value"""
        return self.params.get("Cm")

    @property
    def toff(self) -> float:
        return self.params.get("toff")

    def E_L(self, t=None):
        if t is None:
            t = self.t
        return l(0.025 * np.power((t - 12.0 - self.d), 2) / (self.eps + self.taup))

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
                np.exp(-0.025 * np.power((self.t - 13 - self.d), 2)) * self.E_L(t=0.7*(27-self.t+self.d))) / 2.0

    @property
    def I_S(self):
        return 1.0 - 0.23 * self.c - 0.97 * self.m

    @property
    def I_E(self):
        return self.a * self.I_S

    @property
    def G(self):
        return vectorized_G(self.t, self.I_E, self.tM, self.taug, self.B, self.Cm, self.toff)

    @property
    def g(self):
        """g: get the per-meal post-prandial glucose dynamics

        Examples:
        ---------
            >>> cma = CMASleepWakeModel(N=10, taug=[1.0, 2.0, 3.0])
            >>> cma.g
            array([[0.1       , 0.1       , 0.1       , 0.67372597, 0.11578822,
                    0.10127545, 0.1002057 , 0.10005099, 0.10001504, 0.10000496],
                   [0.1       , 0.1       , 0.1       , 0.1       , 0.1       ,
                    0.88518704, 0.47741404, 0.27071864, 0.17879965, 0.13754265],
                   [0.1       , 0.1       , 0.1       , 0.1       , 0.1       ,
                    0.1       , 0.1       , 0.26789482, 1.2392332 , 1.1899635 ]])

        """
        return self.G

    def integrate_signal(self, signal: np.ndarray | None = None, signal_name: str | None = None,
                         t0: int | float = 0, t1: int | float = 24, M: int = 3,
                         t_extra: Tuple | None = None, tvec: np.ndarray | None = None):
        """integrate the signal between the hours given, assuming M discrete events.

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
        """compute the total morning integrated signal"""
        return self.integrate_signal(signal=signal, signal_name=signal_name, t0=4, t1=13)

    def evening(self, signal: np.ndarray = None, signal_name=None):
        """compute the total evening integrated signal"""
        return self.integrate_signal(signal=signal, signal_name=signal_name, t0=16, t1=24, t_extra=(0, 3))

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
        """vector of instantaneous (overall) glucose"""
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
                                  t      c      m      a    I_S    I_E      L    g_0    g_1    g_2      G  is_meal
            ---------------  ------  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ---------
            0 days 00:00:00   0.000  0.083  0.854  0.251  0.153  0.038  0.000  0.100  0.100  0.100  0.300  False
            0 days 08:00:00   8.000  0.962  0.003  0.517  0.776  0.401  0.841  0.674  0.100  0.100  0.874  True
            0 days 16:00:00  16.000  0.597  0.000  0.565  0.863  0.488  0.841  0.100  0.104  0.100  0.305  True
            1 days 00:00:00  24.000  0.020  0.854  0.250  0.167  0.042  0.000  0.100  0.100  0.102  0.302  False
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


@dataclass
class CMAPlotConfig:
    """configuration for plotting the CMA model results"""

    plot_cols: Optional[Tuple[str]] = (
        "g_0", "g_1", "g_2", "G", "c", "m", "a", "L", "I_S", "I_E", "is_meal")
    labels: Optional[Tuple[str]] = ("Breakfast", "Lunch", "Dinner",
                                    "Glucose", "Cortisol", "Melatonin",
                                    "Adiponectin", "Photoperiod (irradiance)",
                                    "Insulin (secreted)",
                                    "Insulin (effective)", "Meals")
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
        'k'
    )

    @classmethod
    def get_label(cls, col: Container | AnyStr):
        if not isinstance(col, str):
            return [cls.get_label(c) for c in col]
        index = cls.plot_cols.index(col)
        return cls.labels[index]

    @classmethod
    def get_color(cls, col: Container | AnyStr, rgba=False, as_hex=False, keep_alpha=False):
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
    def plot_model_results(cls, df=None, soln=None, plot_cols=None, as_blob=True):
        """plot the results of the model"""
        if df is None:
            df = cls.df
        if soln is None:
            soln = cls.soln
        if plot_cols is None:
            plot_cols = cls.plot_cols
        #: drop is_meal from plot cols... (it's bool afterall)
        plot_cols = list(plot_cols)
        ismeal_ix = plot_cols.index("is_meal")
        plot_cols.pop(ismeal_ix)
        #: combine the data into a single dataframe
        df = df.set_index("t")
        soln = soln.set_index("t")
        df = pd.merge_ordered(df.copy(), soln, suffixes=("", "_soln"), on="t")
        df = df.set_index("t")
        fig, axs = plt.subplots(nrows=2)
        ax = df.plot(ax=axs[0], y="G", color='tab:orange', linestyle='',
                     marker='o', markersize=4, label="rel glucose (data)")
        ax1 = ax.twinx()
        ax1.set_yticks([df["G"].min(), df["G"].max()], [
                       df['value'].min(), df['value'].max()])

        ax2 = df.plot.area(y=plot_cols, color=cls.get_color(plot_cols), ax=axs[1],
                           alpha=0.2, label=cls.get_label(plot_cols), stacked=True)
        ax = df.plot(y="G_soln", color='k', ax=ax, label="rel glucose (model)")
        ax.vlines(x=df.loc[df.is_meal].index,
                  ymin=ax.get_ylim()[0], ymax=df.G_soln.max(),
                  color='r', lw=3, linestyle='--', label='estimated mealtimes')
        ax.legend()
        #: return the figure and axes (unless this is to be a blob for the web)
        if as_blob is False:
            return fig, axs
        bio = BytesIO()
        fig.savefig(bio, format='png')
        bio.seek(0)
        bytes_value = bio.getvalue()
        img_src = 'data:image/png;base64,'
        img_src = img_src + b64encode(bytes_value).decode('utf-8')
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
        #: ref: https://docs.pydantic.dev/latest/usage/exporting_models/#json_encoders
        json_encoders = {
            CMASleepWakeModel: str,
            pd.DataFrame: lambda df: df.to_json(),
            np.ndarray: lambda arr: arr.tolist(),
        }


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
        "max_nfev": data[ycol].size * 400,
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
        n_meals = int(kwds.pop("n_meals", 4))
        df = data[['t', ycol]]
        if not isinstance(df.index, pd.TimedeltaIndex):
            df['dt'] = pd.to_timedelta(df["t"], "H")
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
    p0_cma = dict(zip(pkeys_include, popt))
    cma = cma.update(inplace=False, **p0_cma)

    return CMAFitResult(soln=cma.df, cma=cma, popt=popt, pcov=pcov, infodict=infodict, mesg=mesg, ier=ier)


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
        df, ycol="G", tm_freq="15T", curve_fit_kwds=curve_fit_kwds, taup=taup_est)
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
        ax.get_figure().savefig("./test_fit_cma_model.png")
    return fit_result


@click.group()
@click.pass_context
def cli(ctx):
    pass


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


fit_result_global = None


@cli.command()
@click.option("--N", default=288, type=click.INT)
@click.option("--plot/--no-plot", is_flag=True, default=False)
@click.option("--opts", "--curve-fit-kwds", multiple=True, type=click.Tuple([str, click.UNPROCESSED]),
              callback=process_kwds)
@click.option("--model-config", "--config", prompt=True, default="{}", type=str)
@click.pass_context
def run_fit_model(ctx, n, plot, opts, model_config):
    global fit_result_global
    model_config = json.loads(model_config)
    fit_result = test_fit_model(n=n, plot=plot, opts=opts, **model_config)
    fit_result_global = fit_result
    if plot is True:
        click.confirm("[enter] to exit...", default=True,
                      abort=True, show_default=False)


@cli.command()
@click.pass_context
def run_param_grid(ctx):
    global fit_result_global
    fit_result_global = []
    keys = list(CMASleepWakeModel.param_keys)
    lb = list(CMASleepWakeModel.bounds.lb)
    ub = list(CMASleepWakeModel.bounds.ub)
    tmK = ["tM0", "tM1", "tM2"]
    tmL, tmU = [0, 11, 13], [13, 17, 24]
    plist = list(zip(keys, lb, ub))
    pdict = {}
    pdict = {"tM0": [7, ], "tM1": [12, ], "tM2": [
        18, ], "d": [-3.0, -2.0, 0.0, 1.0, 2.0], }
    # pdict = {k: np.linspace(l, u, num=3) for k, l, u in plist}
    # pdict.update({k: list(range(l, u, 3)) for k, l, u in zip(tmK, tmL, tmU)})
    pgrid = ParameterGrid(pdict)
    cma = CMASleepWakeModel(N=48)
    for i, params in enumerate(pgrid):
        print(f"Iteration ({i:03d}/{len(pgrid)}) ...")
        tM = [params.pop(tmk) for tmk in tmK]
        params["tM"] = tM
        cma.update(**params)
        out = cma.run()
        fit_result_global.append([params, out])
    print('...done.')


@cli.command()
def run_doctests():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    cli()
