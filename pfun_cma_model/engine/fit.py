from pfun_cma_model.engine.data_utils import (
    dt_to_decimal_hours, format_data, downsample_data
)
from pfun_cma_model.engine.cma import CMASleepWakeModel
from pfun_cma_model.misc.types import NumpyArray
import pandas as pd
from typing import Annotated, Any, Dict, Iterable, Container, Generator
import numpy as np
from pydantic import BaseModel, computed_field, ConfigDict, field_serializer  # type: ignore
import importlib
import sys
from pathlib import Path
import logging
from scipy.optimize import minimize  # type: ignore
from scipy.optimize._optimize import OptimizeResult  # type: ignore

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

# import custom cma model

# import custom data utils


class CMAFitResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    soln: pd.DataFrame
    formatted_data: pd.DataFrame
    cma: Any
    popt: Annotated[np.ndarray, NumpyArray] | np.ndarray
    pcov: Annotated[np.ndarray, NumpyArray] | np.ndarray
    infodict: Dict
    mesg: str
    ier: int

    def model_dump_json(
        self,
        *,
        indent=None,
        include=None,
        # exclude infodict (@ v0.3.2a1 fails to serialize on nested numpy arrays)
        exclude=['infodict'],
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
            elif isinstance(value, (Generator)):
                logging.warning(
                    "Could not convert '%s' (key=%s, type=%s) to JSON (saving as naive string representation).",
                    str(key), str(value), type(value))
                self.__dict__[key] = str(value)
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
    def cond(self) -> float:
        cond = np.linalg.cond(self.pcov)
        return cond

    @computed_field
    def diag(self) -> np.ndarray:
        pcov = self.pcov
        return np.diag(pcov)

    @field_serializer("soln", "formatted_data")
    def serialize_dataframe(self, df: pd.DataFrame | Dict, *args) -> dict:
        if isinstance(df, pd.DataFrame):
            return pd.json_normalize(df.to_dict()).to_dict()
        return df

    @field_serializer("popt", "pcov", "diag")
    def serialize_numpy_array(self, arr: Annotated[np.ndarray, NumpyArray] | np.ndarray | list, *args) -> list:
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
        # ! 'h' to indicate hours ('H' is deprecated)
        df = df.assign(dt=pd.to_timedelta(df["t"], "h"))
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
        self.errors: dict[int, list[Any]] = {}
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
    ftol = kwds.get("ftol", 1.49012e-16)
    xtol = kwds.get("xtol", 1.49012e-16)
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
    bounded_methods = [mm.lower() for mm in ["L-BFGS-B",
                                             "Nelder-Mead", "Powell", "TNC", "trust-constr"]]
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
        raise RuntimeError(
            f"Optimal parameters not found: {errmsg}\n{result.message}")
    return popt, pcov, infodict, errmsg, ier


def fit_model(
    data: pd.DataFrame | Dict,
    tcol: str = "t",
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
    - tcol (str) : name of timestamp index column)
    - ycol (str) : name of output data column
    - tM (optional) : vector of mealtimes (decimal hours).
        If unspecified, mealtimes will be estimated (default).
    """
    if curve_fit_kwds is None:
        curve_fit_kwds = {}

    # N takes precedence if passed explicitly in kwds
    if 'N' in kwds and 'n' in kwds:
        raise ValueError("Cannot specify both 'N' and 'n' in kwargs.")
    # N timestamps for final result (data is downsampled internally)
    N = kwds.pop("N", kwds.pop('n', 1024))

    # pre-process data to ensure it is in the correct format
    data = format_data(data, N=N)  # reformat columns, downsample as needed

    # configure curve_fit kwargs
    # set maximum number of function evaluations
    max_nfev = data[ycol].size * 1000
    default_cf_kwds = {
        "verbose": 0,
        "ftol": 1e-18,
        "xtol": 1e-18,
        "max_nfev": max_nfev,
        "method": "L-BFGS-B",
    }
    default_cf_kwds.update(curve_fit_kwds)
    curve_fit_kwds = dict(default_cf_kwds)

    # ! Important to set xdata after 'format_data' has already been called
    # ! ...to ensure the index is of the correct size
    xdata = data[tcol].to_numpy(dtype=float)
    ydata = data[ycol].to_numpy(dtype=float, na_value=np.nan)

    if tM is None:
        tM = estimate_mealtimes(data, ycol, tm_freq=tm_freq, **kwds)

    # ensure there is only one value for t (should not be in kwds)
    t_kwds = kwds.pop("t", 0)
    if t_kwds == 0:
        t_kwds = xdata  # use xdata if no other value for t is given
    t = t_kwds
    if t_kwds is not xdata:
        logger.warning(
            "Two values were provided for 't' parameter... Using: '%s'",
            str(t))
    cma = CMASleepWakeModel(t=t, N=t.size, tM=tM, **kwds)
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
