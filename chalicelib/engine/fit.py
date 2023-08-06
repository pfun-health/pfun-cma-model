from scipy.optimize import curve_fit
import pandas as pd
from typing import Any, Optional, Dict, Iterable
import numpy as np
from pydantic import BaseModel, validator
import importlib
dt_to_decimal_hours = importlib.import_module(
        ".data_utils", package="").dt_to_decimal_hours


class CMAFitResult(BaseModel):
    soln: pd.DataFrame
    cma: Any
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
