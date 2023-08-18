import warnings
from minpack import lmdif
import pandas as pd
from typing import Any, Optional, Dict, Iterable, Container
import numpy as np
from numpy.linalg import LinAlgError
from pydantic import BaseModel, computed_field
import importlib
import sys
from pathlib import Path

root_path = str(Path(__file__).parents[1])
mod_path = str(Path(__file__).parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if mod_path not in sys.path:
    sys.path.insert(0, mod_path)

CMASleepWakeModel = importlib.import_module(
    ".cma_sleepwake", package="chalicelib.engine").CMASleepWakeModel
dt_to_decimal_hours = importlib.import_module(
    ".data_utils", package="chalicelib.engine").dt_to_decimal_hours
format_data = importlib.import_module(
    ".data_utils", package="chalicelib.engine").format_data


class CMAFitResult(BaseModel, arbitrary_types_allowed=True):
    soln: pd.DataFrame
    formatted_data: pd.DataFrame
    cma: Any
    popt: np.ndarray
    pcov: np.ndarray
    infodict: Dict
    mesg: str
    ier: int

    @computed_field
    @property
    def popt_named(self) -> Dict:
        return {k: v for k, v in zip(self.cma.param_keys, self.popt, strict=True)}

    @computed_field
    @property
    def cond(self) -> float:
        #: compute the condition number of the covariance matrix
        #: Note: this should be small if all params are needed
        #: ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit
        cond = np.linalg.cond(self.pcov)
        return cond

    @computed_field
    @property
    def diag(self) -> np.ndarray:
        #: compute the diagonal of the covariance matrix
        #: NOTE: these should be small if all params are needed
        pcov = self.pcov
        return np.diag(pcov)


def estimate_mealtimes(data, ycol: str = 'G', tm_freq: str = "2h",
                       n_meals: int = 4, **kwds):
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


class CurveFitNS:
    """Curve fit namespace.
    """

    LEASTSQ_SUCCESS = [1, 2, 3, 4]
    LEASTSQ_FAILURE = [5, 6, 7, 8]

    def __init__(self, xtol, ftol, maxfev, gtol) -> None:
        self.xtol, self.ftol, self.maxfev, self.gtol = xtol, ftol, maxfev, gtol
        self.errors = {}
        self.get_errors()

    def get_errors(self):
        """
        Get the errors associated with the optimization process.

        Returns:
            dict: A dictionary containing error codes and their corresponding error messages.
                  The keys are integer error codes, and the values are lists with two elements.
                  The first element is a string describing the error, and the second element
                  is the type of error (TypeError or ValueError). If the error type is None,
                  it means that the error is not associated with a specific error type.
        """
        self.errors = {
            0: ["Improper input parameters."],
            1: [f"Both actual and predicted relative reductions in the sum of squares are at most {self.ftol}"],
            2: [f"The relative error between two consecutive iterates is at most {self.xtol}"],
            3: [f"Both actual and predicted relative reductions in the sum of squares are at most {self.ftol} "
                f"and the relative error between two consecutive iterates is at most {self.xtol}"],
            4: [f"The cosine of the angle between func(x) and any column of the Jacobian is at most {self.gtol} "
                f"in absolute value"],
            5: [f"Number of calls to function has reached maxfev = {self.maxfev}"],
            6: [f"ftol={self.ftol} is too small, no further reduction in the sum of squares is possible."],
            7: [f"xtol={self.xtol} is too small, no further improvement in the approximate solution is possible."],
            8: [f"gtol={self.gtol} is too small, func(x) is orthogonal to the columns of the Jacobian to machine precision."]
        }

        return self.errors


def curve_fit(fun, xdata, ydata, p0=None, bounds=None,
              **kwds):
    ftol = kwds.get('ftol', 1.49012e-8)
    xtol = kwds.get('xtol', 1.49012e-8)
    gtol = kwds.get('gtol', 0.0)
    maxfev = kwds.get('max_nfev', 148000)
    cns = CurveFitNS(xtol, ftol, maxfev, gtol)
    fvec = np.zeros(len(xdata), dtype=np.float64)
    p0 = np.array(p0).flatten()
    pcov = np.eye(len(p0), dtype=np.float64)
    pmu = np.eye(len(p0), dtype=np.float64)
    Niters = np.zeros(1, dtype=np.int64)
    diag = np.ones(len(p0), dtype=np.dtype("f8"))
    ier = lmdif(fun, p0, fvec, args=(ydata, pcov, pmu, Niters),
                xtol=xtol, gtol=gtol, maxfev=maxfev, diag=diag)
    popt = p0.copy()
    errmsg, err = cns.errors.get(ier)
    infodict = {"message": errmsg, "error": err, "ier": ier}
    if ier not in cns.LEASTSQ_SUCCESS:
        raise RuntimeError("Optimal parameters not found: " + errmsg)
    return popt, pcov, infodict, errmsg, ier


def fit_model(data: pd.DataFrame | Dict, ycol: str = "G",
              tM: None | Iterable = None, tm_freq: str = "2h",
              curve_fit_kwds: Dict = {}, **kwds) -> CMAFitResult:
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

    data = format_data(data)

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
        tM = estimate_mealtimes(data, ycol, tm_freq=tm_freq, **kwds)

    #: instantiate model
    cma = CMASleepWakeModel(t=xdata, N=None, tM=tM, **kwds)
    if curve_fit_kwds.get("verbose"):
        print("taup0=", cma.taup)

    def fun(p, fvec, args=(), cma=cma):
        y, pcov, pmu, Niters = args
        if pmu is not None:
            pmu[:] = ((p + pmu) / 2.0)[:]
        if pcov is not None:
            pcov[:, :] = ((p - pmu)*(p - pmu).T / (Niters + 1))[:, :]
        d, taup, taug, B, Cm, toff = p
        cma.update(inplace=True, d=d, taup=taup,
                   taug=taug, B=B, Cm=Cm, toff=toff)
        fvec[:] = np.power(y - cma.g_instant, 2)[:]
        if Niters is not None:
            Niters[0] = Niters[0] + 1

    #: perform fitting
    pkeys_include = cma.param_keys
    p0 = np.array([cma.params[k] for k in pkeys_include])
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
                        infodict=infodict, mesg=mesg, ier=ier,
                        formatted_data=data)
