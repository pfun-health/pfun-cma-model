import pandas as pd
from typing import Any, Dict, Iterable, Container
import numpy as np
from pydantic import BaseModel, computed_field, ConfigDict, field_serializer
import importlib
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

root_path = str(Path(__file__).parents[1])
mod_path = str(Path(__file__).parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if mod_path not in sys.path:
    sys.path.insert(0, mod_path)

CMASleepWakeModel = importlib.import_module(
    ".cma_sleepwake", package="pfun_cma_model.runtime.src.engine").CMASleepWakeModel
dt_to_decimal_hours = importlib.import_module(
    ".data_utils", package="pfun_cma_model.runtime.src.engine").dt_to_decimal_hours
format_data = importlib.import_module(".data_utils",
                                      package="pfun_cma_model.runtime.src.engine").format_data

try:
    lmdif = importlib.import_module(
        'minpack'
    ).lmdif
except ImportError:
    try:
        import ctypes
        libpath = os.path.expanduser('~/.local/lib')
        ctypes.cdll.LoadLibrary(os.path.join(libpath, 'libminpack.so'))
        lmdif = importlib.import_module(
            'minpack'
        ).lmdif
    except Exception:
        logger.warning("Failed to load minpack library.", exc_info=True)


class CMAFitResult(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict()
    soln: pd.DataFrame
    formatted_data: pd.DataFrame
    cma: Any
    popt: np.ndarray
    pcov: np.ndarray
    infodict: Dict
    mesg: str
    ier: int

    def model_dump_json(
        self,
        *,
        indent=None,
        include=None,
        exclude=None,
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

    @field_serializer("soln", "formatted_data")
    def serialize_dataframe(self, df: pd.DataFrame | Dict, *args) -> dict:
        if isinstance(df, pd.DataFrame):
            return pd.json_normalize(df.to_dict()).to_dict()
        return df

    @field_serializer("popt", "pcov", "diag")
    def serialize_numpy_array(self, arr: np.ndarray | list, *args) -> list:
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        return arr

    @field_serializer("cma")
    def serialize_cma(self, cma: Any, _info):
        if hasattr(cma, "to_dict"):
            return cma.to_dict()
        return cma


def estimate_mealtimes(data,
                       ycol: str = "G",
                       tm_freq: str = "2h",
                       n_meals: int = 4,
                       **kwds):
    n_meals = int(n_meals)
    df = data[["t", ycol]]
    if not isinstance(df.index, pd.TimedeltaIndex):
        df = df.assign(dt=pd.to_timedelta(df["t"], "H"))
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
            0: ["Improper input parameters.", TypeError],
            1: [
                "Both actual and predicted relative reductions "
                "in the sum of squares\n  are at most %f" % self.ftol,
                None,
            ],
            2: [
                "The relative error between two consecutive "
                "iterates is at most %f" % self.xtol,
                None,
            ],
            3: [
                "Both actual and predicted relative reductions in "
                "the sum of squares\n  are at most {:f} and the "
                "relative error between two consecutive "
                "iterates is at \n  most {:f}".format(self.ftol, self.xtol),
                None,
            ],
            4: [
                "The cosine of the angle between func(x) and any "
                "column of the\n  Jacobian is at most %f in "
                "absolute value" % self.gtol,
                None,
            ],
            5: [
                "Number of calls to function has reached "
                "maxfev = %d." % self.maxfev,
                ValueError,
            ],
            6: [
                "ftol=%f is too small, no further reduction "
                "in the sum of squares\n  is possible." % self.ftol,
                ValueError,
            ],
            7: [
                "xtol=%f is too small, no further improvement in "
                "the approximate\n  solution is possible." % self.xtol,
                ValueError,
            ],
            8: [
                "gtol=%f is too small, func(x) is orthogonal to the "
                "columns of\n  the Jacobian to machine "
                "precision." % self.gtol,
                ValueError,
            ],
        }

        return self.errors


def curve_fit(fun, xdata, ydata, p0=None, bounds=None, **kwds):
    """
    Curve fitting function that estimates the optimal parameters for a given function based on input data.

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
    gtol = kwds.get("gtol", 0.0)
    maxfev = kwds.get("max_nfev", 150000)
    cns = CurveFitNS(xtol, ftol, maxfev, gtol)
    fvec = np.zeros(len(xdata), dtype=np.float64)
    p0 = np.array(p0).flatten()
    pcov = np.eye(len(p0), dtype=np.float64)
    pmu = np.eye(len(p0), dtype=np.float64)
    Niters = np.zeros(1, dtype=np.int64)
    diag = np.ones(len(p0), dtype=np.dtype("f8"))
    ier = lmdif(
        fun,
        p0,
        fvec,
        args=(ydata, pcov, pmu, Niters),
        xtol=xtol,
        gtol=gtol,
        maxfev=maxfev,
        diag=diag,
    )
    popt = p0.copy()
    errout = cns.errors[ier]
    if len(errout) == 2:
        errmsg, err = errout
    else:
        err = None
        errmsg = errout
    infodict = {"message": errmsg, "error": err, "ier": ier}
    if ier not in cns.LEASTSQ_SUCCESS:
        raise RuntimeError(f"Optimal parameters not found: {errmsg}")
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
    #: handle curve_fit_kwds missing
    if curve_fit_kwds is None:
        curve_fit_kwds = {}

    #: format data
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
        logging.debug("taup0=%f", cma.taup)

    def fun(p, fvec, args=(), cma=cma):
        y, pcov, pmu, Niters = args
        if pmu is not None:
            pmu[:] = ((p + pmu) / 2.0)[:]
        if pcov is not None:
            pcov[:, :] = ((p - pmu) * (p - pmu).T / (Niters + 1))[:, :]
        d, taup, taug, B, Cm, toff = p
        cma.update(inplace=True,
                   d=d,
                   taup=taup,
                   taug=taug,
                   B=B,
                   Cm=Cm,
                   toff=toff)
        fvec[:] = np.power(y - cma.g_instant, 2)[:]
        if Niters is not None:
            Niters[0] = Niters[0] + 1

    #: perform fitting
    pkeys_include = cma.bounded_param_keys
    p0 = np.array([cma.params[k] for k in pkeys_include])
    if isinstance(p0[cma.bounded_param_keys.index("taug")], Container):
        #: ! ensure we update using a scalar
        p0[cma.bounded_param_keys.index("taug")] = 1.0
    bounds = cma.bounds
    popt_internal, pcov, infodict, mesg, ier = curve_fit(fun,
                                                xdata,
                                                ydata,
                                                p0=p0,
                                                bounds=bounds,
                                                full_output=True,
                                                **curve_fit_kwds)
    popt = np.array([cma.params[k] for k in pkeys_include])

    #: informed model (best fit)
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
