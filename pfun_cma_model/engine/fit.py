import pandas as pd
from typing import Any, Dict, Iterable, Container
import numpy as np
from pydantic import BaseModel, computed_field, ConfigDict, field_serializer
import importlib
import sys
import os
from pathlib import Path
import logging
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

root_path = str(Path(__file__).parents[1])
mod_path = str(Path(__file__).parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if mod_path not in sys.path:
    sys.path.insert(0, mod_path)

__all__ = [
    "CMAFitResult",
    "fit_model",
    "estimate_mealtimes"
]

# import custom ndarray schema
NumpyArray = importlib.import_module(
    ".types", package="pfun_cma_model.misc").NumpyArray

# import custom cma model
CMASleepWakeModel = importlib.import_module(
    ".cma", package="pfun_cma_model.engine").CMASleepWakeModel

# import custom data utils
dt_to_decimal_hours = importlib.import_module(
    ".data_utils", package="pfun_cma_model.engine").dt_to_decimal_hours
format_data = importlib.import_module(".data_utils",
                                      package="pfun_cma_model.engine").format_data


class CMAFitResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    soln: pd.DataFrame
    formatted_data: pd.DataFrame
    cma: Any
    popt: NumpyArray | np.ndarray
    pcov: NumpyArray | np.ndarray
    infodict: Dict
    mesg: str
    ier: int

    def model_dump_json(
        self,
        *,
        indent=None,
        include=None,
        exclude=['infodict'],  # exclude infodict (@ v0.3.2a1 fails to serialize on nested numpy arrays)
        by_alias=False,
        exclude_unset=False,
        exclude_defaults=False,
        exclude_none=False,
        round_trip=False,
        warnings=True,
    ):
        original_dict = self.__dict__.copy()
        for key, value in self.__dict__.items():
            if isinstance(value, pd.DataFrame):
                self.__dict__[key] = value.to_json()
            if isinstance(value, np.ndarray):
                self.__dict__[key] = value.tolist()
            if isinstance(value, CMASleepWakeModel):
                self.__dict__[key] = value.dict()  # type: ignore
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, OptimizeResult):
                        value[k] = v.__repr__()  # handle OptimizeResult
                    if isinstance(v, pd.DataFrame):
                        value[k] = v.to_json()
                    if isinstance(v, np.ndarray):
                        value[k] = v.tolist()
                    if isinstance(v, CMASleepWakeModel):
                        value[k] = v.dict()  # type: ignore
                self.__dict__[key] = value
        try:
            output = super().model_dump_json(
                indent=indent,
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                round_trip=round_trip,
                warnings=warnings,
            )
        except Exception as error:
            logging.warning("Failed to dump model json.", exc_info=True)
            for key in self.__dict__:
                value = self.__dict__[key]
                logging.info(key, type(value), type(self.__dict__[key]))
                print(key, type(value), type(self.__dict__[key]))
            raise error
        self.__dict__.update(original_dict)
        return output

    @computed_field
    @property
    def popt_named(self) -> Dict:
        if hasattr(self.cma, "bounded_param_keys"):
            bounded_param_keys = self.cma.bounded_param_keys
        else:
            bounded_param_keys = self.cma.get("bounded_param_keys")
        return {
            k: v
            for k, v in zip(bounded_param_keys, self.popt, strict=True)
        }

    @computed_field
    @property
    def cond(self) -> float:
        cond = np.linalg.cond(self.pcov)
        return cond

    @computed_field
    @property
    def diag(self) -> np.ndarray:
        pcov = self.pcov
        return np.diag(pcov)

    @field_serializer("soln", "formatted_data")
    def serialize_dataframe(self, df: pd.DataFrame | Dict, *args) -> dict:
        if isinstance(df, pd.DataFrame):
            return pd.json_normalize(df.to_dict()).to_dict()
        return df

    @field_serializer("popt", "pcov", "diag")
    def serialize_numpy_array(self, arr: NumpyArray | np.ndarray | list, *args) -> list:
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        return arr

    @field_serializer("cma")
    def serialize_cma(self, cma: Any, _info):
        if hasattr(cma, "to_dict"):
            return cma.to_dict()
        return cma
    
    @field_serializer("infodict")
    def serialize_infodict(self, infodict: Dict, _info):
        if isinstance(infodict, OptimizeResult):
            return infodict.__repr__()
        if isinstance(infodict, dict):
            for k, v in infodict.items():
                if isinstance(v, OptimizeResult):
                    infodict[k] = v.__repr__()
        return infodict


def estimate_mealtimes(data,
                       ycol: str = "G",
                       tm_freq: str = "2h",
                       n_meals: int = 4,
                       **kwds):
    n_meals = int(n_meals)
    df = data[["t", ycol]]
    if not isinstance(df.index, pd.TimedeltaIndex):
        df = df.assign(dt=pd.to_timedelta(df["t"], "h"))  # ! 'h' to indicate hours ('H' is deprecated)
        df.set_index("dt", inplace=True)
    dfres = df.resample(tm_freq).mean()
    tM = (dfres[ycol].diff().dropna().groupby(
        pd.Grouper(freq=tm_freq)).max().sort_values().index.to_series().apply(
            lambda d: dt_to_decimal_hours(d)).unique()[-n_meals:] - 0.05)
    tM[tM < 0.0] += 23.9999
    tM[tM > 24.0] -= 23.9999
    tM.sort()
    return tM


class CurveFitNS:
    """Curve fit namespace."""

    LEASTSQ_SUCCESS = [0]
    LEASTSQ_FAILURE = [1, 2, 3, 4]

    def __init__(self, xtol, ftol, maxfev, gtol) -> None:
        self.xtol, self.ftol, self.maxfev, self.gtol = xtol, ftol, maxfev, gtol
        self.errors = {}
        self.get_errors()

    def get_errors(self):
        self.errors = {
            0: ["Optimization terminated successfully.", None],
            1: ["Maximum number of function evaluations has been exceeded.", ValueError],
            2: ["Optimization failed to converge.", ValueError],
            3: ["Optimization stopped for unknown reason.", ValueError],
            4: ["Optimization failed due to error.", ValueError],
        }
        return self.errors


def curve_fit(fun, xdata, ydata, p0=None, bounds=None, **kwds):
    """
    Curve fitting using scipy.optimize.minimize.

    Parameters:
        fun: callable
            The function to be fitted.
        xdata: array-like
            The input data for the independent variable.
        ydata: array-like
            The input data for the dependent variable.
        p0: array-like, optional
            Initial guess for the parameters.
        bounds: tuple or list, optional
            Bounds on parameters.
        **kwds: dict
            Additional keyword arguments.

    Returns:
        popt: array-like
            Optimal values for the parameters.
        pcov: 2-D array
            Estimated covariance of popt.
        infodict: dict
            A dictionary containing additional information.
        errmsg: str
            Error message, if any.
        ier: int
            An integer flag indicating the convergence status.

    Raises:
        RuntimeError: If optimal parameters are not found.
    """
    ftol = kwds.get("ftol", 1.49012e-8)
    xtol = kwds.get("xtol", 1.49012e-8)
    maxfev = kwds.get("max_nfev", 150000)
    method = kwds.get("method", "L-BFGS-B")
    cns = CurveFitNS(xtol, ftol, maxfev, 0.0)

    def obj_func(p):
        fvec = np.zeros_like(ydata)
        fun(p, fvec, args=(ydata, None, None, None))
        return np.sum(fvec)

    options = {
        "ftol": ftol,
        "maxfun": maxfev,
        "maxiter": maxfev,
        "disp": kwds.get("verbose", 0) > 0,
    }
    # handle method-specific options (bounded/unbounded)
    bounded_methods = [mm.lower() for mm in ["L-BFGS-B", "Nelder-Mead", "Powell", "TNC", "trust-constr"]]
    if method.lower() not in bounded_methods:
        options.pop("maxfun", None)
    # Convert bounds to scipy format if needed
    if bounds is not None:
        if isinstance(bounds, dict):
            bounds = [tuple(bounds[k]) for k in sorted(bounds)]
        elif isinstance(bounds, (list, tuple)) and isinstance(bounds[0], (list, tuple, np.ndarray)):
            bounds = [tuple(b) for b in bounds]
        else:
            bounds = None

    result = minimize(
        obj_func,
        p0,
        method=method,
        bounds=bounds,
        options=options,
    )

    popt = result.x
    # Estimate covariance matrix if possible
    if result.hess_inv is not None and hasattr(result.hess_inv, 'todense'):
        pcov = result.hess_inv.todense()
    elif result.hess_inv is not None and hasattr(result.hess_inv, '__array__'):
        pcov = np.atleast_2d(result.hess_inv)
    else:
        pcov = np.eye(len(popt))

    ier = 0 if result.success else 2
    errmsg, err = cns.errors.get(ier, ["Unknown error", None])
    infodict = {"message": errmsg, "error": err, "ier": ier, "result": result}
    if not result.success:
        raise RuntimeError(f"Optimal parameters not found: {errmsg}\n{result.message}")
    return popt, pcov, infodict, errmsg, ier


def fit_model(
    data: pd.DataFrame | Dict,
    ycol: str = "G",
    tM: None | Iterable = None,
    tm_freq: str = "2h",
    curve_fit_kwds: Dict | None = None,
    **kwds,
) -> CMAFitResult:
    """use `curve_fit` to fit the model to data

    Arguments:
    ----------
    - data (pd.DataFrame) : ["t", "ycol"]
        "t"      : 24-hour hour of day
        "<ycol>" : raw egv
    - ycol (str) : name of output data column
    - tM (optional) : vector of mealtimes (decimal hours).
        If unspecified, mealtimes will be estimated (default).
    """
    if curve_fit_kwds is None:
        curve_fit_kwds = {}

    # pre-process data to ensure it is in the correct format
    data = format_data(data)

    # configure curve_fit kwargs
    max_nfev = data[ycol].size * 500  # set maximum number of function evaluations
    default_cf_kwds = {
        "verbose": 0,
        "ftol": 1e-6,
        "xtol": 1e-6,
        "max_nfev": max_nfev,
        "method": "L-BFGS-B",
    }
    default_cf_kwds.update(curve_fit_kwds)
    curve_fit_kwds = dict(default_cf_kwds)

    xdata = data["t"].to_numpy(dtype=float)
    ydata = data[ycol].to_numpy(dtype=float, na_value=np.nan)

    if tM is None:
        tM = estimate_mealtimes(data, ycol, tm_freq=tm_freq, **kwds)

    if 'N' in kwds and 'n' in kwds:
        raise ValueError("Cannot specify both 'N' and 'n' in kwargs.")
    N = kwds.pop("N", kwds.pop('n', None))  # default N timesteps to class default
    cma = CMASleepWakeModel(t=xdata, N=N, tM=tM, **kwds)
    if curve_fit_kwds.get("verbose"):
        logging.debug("taup0=%f", cma.taup)

    def fun(p, fvec, args=(), cma=cma):
        y, pcov, pmu, Niters = args
        d, taup, taug, B, Cm, toff = p
        cma.update(inplace=True,
                   d=d,
                   taup=taup,
                   taug=taug,
                   B=B,
                   Cm=Cm,
                   toff=toff)
        fvec[:] = np.power(y - cma.g_instant, 2)[:]

    pkeys_include = cma.bounded_param_keys
    p0 = np.array([cma.params[k] for k in pkeys_include])
    if isinstance(p0[cma.bounded_param_keys.index("taug")], Container):
        p0[cma.bounded_param_keys.index("taug")] = 1.0
    bounds = cma.bounds

    popt_internal, pcov, infodict, mesg, ier = curve_fit(
        fun,
        xdata,
        ydata,
        p0=p0,
        bounds=bounds,
        **curve_fit_kwds
    )
    popt = popt_internal

    p0_cma = dict(zip(pkeys_include, popt, strict=True))
    cma = cma.update(inplace=False, **p0_cma)

    return CMAFitResult(
        soln=cma.df,
        cma=cma,
        popt=popt,
        pcov=pcov,
        infodict=infodict,
        mesg=mesg,  # type: ignore
        ier=ier,  # type: ignore
        formatted_data=data,  # type: ignore
    )  # type: ignore
