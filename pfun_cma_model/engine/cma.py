#!/usr/bin/env python
"""app.engine.cma: define the Cortisol-Melatonin-Adiponectin model."""
from pfun_cma_model.misc.types import NumpyArray
from pfun_cma_model.engine.cma_model_params import CMAModelParams, CMABoundedParams
from pfun_cma_model.engine.bounds import Bounds
from pfun_cma_model.engine.calc import exp, Light, E, vectorized_G
from functools import cached_property
import copy
import json
import logging
import sys
from pathlib import Path
from typing import (
    Sequence,
    Iterable,
    Dict,
    Tuple,
    Callable,
    Container,
    Optional,
    Generator,
    Annotated,
)
from numpy import (
    array,
    ndarray,
    nansum,
    power,
    cos,
    pi,
    logical_and,
    logical_or,
    append,
    mod,
    linspace,
    broadcast_to,
    bool_,
    asarray,
)
from numpy import abs as np_abs
from numpy.random import default_rng
from pandas import DataFrame, Series, json_normalize, to_timedelta

#: pfun imports (relative)
root_path = str(Path(__file__).parents[1])
mod_path = str(Path(__file__).parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if mod_path not in sys.path:
    sys.path.insert(0, mod_path)


logger = logging.getLogger()

__all__ = [
    "CMASleepWakeModel",
]


class CMAParamTypeError(TypeError):
    """CMA parameter type error."""

    def __init__(self, pkey, *args: object) -> None:
        super().__init__(*args)
        self._pkey = pkey

    def __repr__(self) -> str:
        return super().__repr__() + f"...Parameter '{self._pkey}' must be numeric."


class CMASleepWakeModel:
    """Defines the Cortisol-Melatonin-Adiponectin Sleep-Wake pfun model.

    Methods:
    -------
    1)) Input SG -> Project SG to 24-hour phase plane.
    2)) Estimate photoperiod (t_m0 - 1, t_m2 + 3) -> Model params (d, taup).
    3)) (Fit to projected SG) Compute approximate chronometabolic dynamics:
        F(m, c, a)(t, d, taup) -> ...
         ...  (+/- postprandial insulin, glucose){Late, Early}.
    """

    @cached_property
    def _DEFAULT_PARAMS_MODEL(self) -> CMAModelParams:  # type: ignore
        return CMAModelParams()

    @cached_property
    def _DEFAULT_PARAMS(self) -> Dict:
        return self._DEFAULT_PARAMS_MODEL.model_dump()

    @cached_property
    def bounded_param_keys(self):
        return self._DEFAULT_PARAMS_MODEL.bounded.bounded_param_keys

    @cached_property
    def param_keys(self):
        return tuple(self._DEFAULT_PARAMS.keys())

    def param_key_index(
        self, keys: str | Iterable[str] | Sequence[str], only_bounded: bool = False
    ) -> int | list[int]:
        """Return the index of the parameter key."""
        local_param_keys = (
            self.param_keys if not only_bounded else self.bounded_param_keys
        )
        if isinstance(keys, str):
            return local_param_keys.index(keys)
        else:
            return [local_param_keys.index(k) for k in keys]

    def update_bounds(
        self,
        keys=[],
        lb=[],
        ub=[],
        keep_feasible: bool_ | Iterable[bool_] = Bounds.True_,
        return_bounds=False,
    ):
        """Update the bounds of the model."""
        keys = [keys] if isinstance(keys, str) else keys
        lb = [float(lb)] if isinstance(lb, (float, int)) else lb
        ub = [float(ub)] if isinstance(ub, (float, int)) else ub
        if isinstance(keep_feasible, bool_):
            keep_feasible = [
                keep_feasible,
            ] * len(keys)
        for k in keys:
            ix = self.bounded_param_keys.index(k)
            self.bounds[ix] = (
                lb[ix], ub[ix], keep_feasible[ix])  # type: ignore
        if return_bounds:
            return self.bounds

    def __json__(self):
        """JSON serialization."""
        out = {k: v for k, v in self.__dict__.items()}
        keys = list(out.keys())
        for key in keys:
            value = out[key]
            if isinstance(value, ndarray):
                out[key] = value.tolist()
            elif isinstance(value, DataFrame):
                out[key] = json_normalize(
                    value.to_dict()).to_dict()  # type: ignore
            elif isinstance(value, Series):
                out[key] = value.tolist()
            elif isinstance(value, Bounds):
                out[key] = value.json()
            elif hasattr(value, "__json__"):
                out[key] = value.__json__()
            elif hasattr(value, "model_dump"):
                out[key] = value.model_dump()
            elif isinstance(value, (Generator)):
                logging.warning(
                    "Could not convert %s (type=%s) to JSON.", str(
                        value), type(value)
                )
                out.pop(key)
            elif isinstance(value, (dict)):
                for k, v in value.items():
                    if isinstance(v, DataFrame):
                        value[k] = v.to_json()
                    elif isinstance(v, ndarray):
                        value[k] = v.tolist()
            try:
                logging.info(
                    "attempting to serialize: %s (value='%s')", key, str(
                        out.get(key))
                )
                json.dumps(out[key])
            except (json.JSONDecodeError, TypeError) as exc:
                logging.warning(
                    "Exception: '%s'\n...Could not convert %s to JSON.",
                    str(exc),
                    str(value),
                    exc_info=False,
                )
                out.pop(key, None)  # ! remove any non-JSONable value
        # ! skipkeys removes all non-basic types
        return json.dumps(out, skipkeys=True)

    def json(self):
        """JSON serialization."""
        return self.__json__()

    def to_dict(self):
        return json.loads(self.__json__())  # type: ignore

    def dict(self):
        return self.to_dict()

    @property
    def params(self) -> CMAModelParams:
        """Return the current parameters as a CMAModelParams object."""
        params_dict = {k: self._params[k]
                       for k in self.param_keys}  # preserve order
        return CMAModelParams(**params_dict)

    # class-level private storage of parameters
    _params: Dict = CMAModelParams().model_dump()

    def new_tvector(self, t0: int | float, t1: int | float, n: int) -> ndarray:
        return self.params.new_tvector(t0, t1, n)

    def __getitem__(self, key):
        """Get parameter value by key."""
        return getattr(self.params, key)

    @params.setter  # type: ignore
    def params(self, value):
        self._params = value

    @property
    def unbounded_param_keys(self):
        return ("t", "N", "tM", "seed", "eps")

    def __init__(self, config: Dict | CMAModelParams | None = None, **kwds):  # type: ignore
        """PFun CMA model constructor.

        Arguments:
        ----------
            t (array or float, optional): Time vector (corresponds to ). If not provided, t will be a linearly spaced vector of length N.
            N (int, optional): Number of time points. Defaults to 24.
            d (float, optional): Offset from UTC solar noon for the estimated latitude (hours). Defaults to 0.0.
            taup (float, optional): Photoperiod length (hours). Defaults to 1.0.
            taug (float, optional): Meal duration (hours). Defaults to 1.0.
            B (float, optional): Bias constant. Defaults to 0.05.
            Cm (float, optional): Cortisol temporal sensitivity coefficient (u/h). Defaults to 0.0.
            toff (float, optional): Solar noon offset for the estimated latitude (hours). Defaults to 0.0.
            tM (tuple, optional): Meal times (hours). Defaults to (7.0, 11.0, 17.5).
            seed (None | int, optional): Random seed value. If provided, random noise will be included in the model solution, scaled by parameter eps. Defaults to None.
            eps (float, optional): Random noise scale ("epsilon"). Defaults to 1e-18.
        """
        # update with any given config:
        if config is not None:
            if isinstance(config, CMAModelParams):
                config = config.model_dump()
            self._params.update(config)
        # update with any given kwds:
        self._params.update(kwds)
        # Setup bounds (for bounded params):
        self.bounds = copy.copy(self._DEFAULT_PARAMS_MODEL.bounds)
        if all(
            [
                kwds.get("lb", False),
                kwds.get("ub", False),
                kwds.get("bounded_param_keys", False),
            ]
        ):
            self.update_bounds(
                kwds["bounded_param_keys"],
                kwds["lb"],
                kwds["ub"],
                kwds.get("keep_feasible", Bounds.True_),
            )
        elif "bounds" in kwds:
            new_bounds = kwds["bounds"]
            if isinstance(new_bounds, str):
                new_bounds = json.loads(new_bounds)
            self.update_bounds(**new_bounds)
        # Setup the random number generator (if seed is given)
        self.rng = None
        if self.seed is not None:
            self.rng = default_rng(seed=self.seed)

    @property
    def eps(self) -> float:
        """eps : float
        Random noise scale ("epsilon").
        """
        return self.params.eps

    @property
    def seed(self) -> Optional[int | float]:
        """seed : Optional[int | float]
        Random seed. Set to an integer to enable random noise via parameter 'eps'. Optional.
        """
        return self.params.seed

    @property
    def tM(self) -> ndarray:
        """tM : ndarray
        Meal times (hours).
        """
        return array(self.params.tM)  # type: ignore

    @tM.setter  # type: ignore
    def tM(self, value):
        self.params.tM = array(value)  # type: ignore

    @property
    def t(self) -> ndarray:
        """t : ndarray
        Time vector (decimal hours).
        """
        return array(self.params.t)

    @t.setter  # type: ignore
    def t(self, value):
        self.params.t = array(value)  # type: ignore

    @property
    def N(self) -> int:
        """N : int
        Number of time points.

        Default to self.t.size.
        """
        return int(self.params.N)

    @N.setter  # type: ignore
    def N(self, value):
        self.params.N = int(value)

    @property
    def bounded_params_as_dict(self) -> Dict:
        """Return the current bounded parameters as a python dict."""
        return {k: self.params.bounded[k] for k in self.bounded_param_keys}

    @property
    def bounded_params(self) -> CMABoundedParams:
        """Return the current bounded parameters as a CMAModelParams object."""
        return self.params.bounded

    @property
    def bounded_params_as_obj(self) -> CMABoundedParams:
        """
        Return the current bounded parameters as a CMAModelParams object.

        Alias method for `self.bounded_params` property.
        """
        return self.bounded_params

    def update_bounded_params(
        self, params: Dict | CMAModelParams | CMABoundedParams
    ) -> CMAModelParams:
        """Update the latest parameters to correspond to the current bounds (trim to bounds).

        Args:
            params (Dict): most recently updated parameters.

        Returns:
            Dict: updated parameters
        """
        # get the bounded parameters from the params dict
        if isinstance(params, CMAModelParams):
            params = params.model_dump()
        # If params is already a dict, do not call model_dump
        bounded_params = {
            k: v for k, v in params.items() if k in self.bounded_param_keys
        }
        bounded_params.update(params.get("bounded", {}))  # type: ignore
        # ! ensure the bounded parameters are within bounds
        new_params = self.bounds.update_values(
            {k: float(v) for k, v in bounded_params.items()}
        )
        self.params.bounded.update(**new_params)
        return self.params

    # type: ignore
    def update(
        self, model_params: Optional[CMAModelParams | Dict] = None, inplace=True, **kwds
    ):
        """
        Update the current instance with new values (occurs inplace by default).

        Parameters:
            *args: Variable length argument list.
            inplace (bool): If True, update the current instance in place. If False, create a new instance with the updated values.
            **kwds: Keyword arguments to update the instance.

        Returns:
            The updated instance if `inplace` is False, otherwise None.

        Raises:
            ValueError: If a parameter is not found.
            TypeError: If a parameter is not numeric.

        Note:
            - If `inplace` is False, a new instance is created with the updated values and returned.
            - Parameters in `kwds` that are not present in the instance's `param_keys` are ignored.
            - The instance's `params` dictionary is updated with the values from `kwds`.
            - The order of the parameters in the `params` dictionary is preserved.
            - The instance's `params` dictionary is updated to keep values within the specified bounds.
            - The `tM` attribute is updated with the values from `kwds` if 'tM' is present.
            - The `t` attribute is updated with a linspace from 0 to 24 if 'N' is present.
            - The `rng` attribute is updated with a new random number generator if 'seed' is present.
            - The `eps` attribute is updated with the value from `kwds` if 'eps' is present.
        """
        if model_params is not None:
            if isinstance(model_params, CMAModelParams):
                model_params = model_params.model_dump()  # convert to dict
            model_params.update(**kwds)  # type: ignore
            kwds = dict(model_params)  # type: ignore
        if inplace is False:
            new_inst = copy.copy(self)
            new_inst.update(inplace=True, **kwds)
            return new_inst
        #: ! handle case in which taug was given as a vector initially
        if "taug" in kwds and isinstance(getattr(self.params, "taug", None), Container):
            taug_new = kwds.pop("taug")
            match isinstance(taug_new, Container):
                case True:
                    #: ! replace current values elementwise if given a vector
                    self.params["taug"] = broadcast_to(  # type: ignore
                        taug_new, (self.n_meals,)
                    )
                case False:  # ! else, taug is a scale: <old_taug> *= new_taug
                    self.params["taug"] = array(  # type: ignore
                        self.params["taug"], dtype=float
                    ) * float(taug_new)
        #: update all given params by updating the private dict directly
        self._params.update(**kwds)
        #: Important next line:
        #: ! Keeps params within specified bounds (keep_feasible is handled by Bounds)
        #: ! Ensures that only bounded params are updated
        self.params = self.update_bounded_params(self.params)  # type: ignore
        if "tM" in kwds:
            # clean up tM (handle string inputs, etc.)
            # ! caution with this (string injection is possible)
            tM_raw = kwds["tM"]
            if isinstance(tM_raw, str):
                # extract numbers from the string
                tM = [
                    float(x)
                    for x in tM_raw.split(",")
                    if x.strip().replace(".", "", 1).isdigit()
                ]
            elif isinstance(tM_raw, (int, float)):
                tM = [tM_raw]
            else:
                try:
                    tM = list(tM_raw)
                except TypeError:
                    raise TypeError(
                        f"Invalid type for tM: {type(tM_raw)}. Must be a list, tuple, or string of numeric values."
                    )
            try:
                tM = [
                    float(x) for x in tM if isinstance(x, (int, float))
                ]  # type: ignore
            except ValueError:
                raise ValueError(
                    f"Invalid value in tM: {tM}. All values must be numeric."
                )
            self.tM = array(tM, dtype=float).flatten()
        if kwds.get("N") is not None:
            self.t = linspace(0, 24, num=int(kwds["N"]))
        if kwds.get("seed") is not None:
            self.rng = default_rng(seed=kwds["seed"])
        if kwds.get("eps") is not None:
            self.eps = kwds["eps"]
        #: check all parameters are present and valid types
        for pkey in self.bounded_param_keys:
            if not isinstance(getattr(self.bounded_params, pkey), (float, int)):
                raise TypeError(f"Parameter '{pkey}' must be numeric.")

    @property
    def d(self) -> float:
        """d : float
        Offset from UTC solar noon for the estimated latitude (hours).
        """
        return self.bounded_params.d

    @property
    def taup(self) -> float:
        """taup : float
        Approximate photoperiod (hours).
        """
        return self.bounded_params.taup

    @property
    def n_meals(self):
        """Number of meals."""
        return len(self.tM)

    @property
    def taug(self) -> ndarray:
        """taug: get an array broadcasted to: (, number_of_meals)."""
        taug_ = self.bounded_params.taug
        taug_vector = broadcast_to(taug_, (self.n_meals,))
        return taug_vector

    @property
    def B(self) -> float:
        """Return the current bias parameter value (B)."""
        return self.bounded_params.B

    @property
    def Cm(self) -> float:
        """return the current Cm param value."""
        return self.bounded_params.Cm

    @property
    def toff(self) -> float:
        return self.bounded_params.toff

    def E_L(self, t=None):
        if t is None:
            t = array(self.t)
        # type: ignore
        return Light(0.025 * power((t - 12.0 - self.d), 2) / (self.eps + self.taup))

    @property
    def L(self):
        return self.E_L(t=self.t)

    def M(self, t=None):
        """compute the estimated relative Melatonin signal."""
        if t is None:
            t = self.t
        m_out = power((1.0 - self.L), 3) * power(
            cos(-(t - 3.0 - self.d) * pi / 24.0), 2
        )  # type: ignore
        if self.rng is not None:
            # ! tiny amount of random noise
            # type: ignore
            m_out = m_out + \
                self.rng.uniform(low=-self.eps, high=self.eps, size=self.N)
        return m_out

    @property
    def m(self):
        return self.M(t=self.t)

    @property
    def c(self):
        return (
            (4.9 / (1.0 + self.taup))
            * pi
            * E(power((self.L - 0.88), 3))
            * E(0.05 * (8.0 - self.t + self.d))
            * E(2.0 * power(-self.m, 3))
        )  # type: ignore

    @property
    def a(self):
        """Compute Adiponectin circadian dynamics.
        """
        return (
            E(power((-self.c * self.m), 3))
            + exp(-0.025 * power((self.t - 13 - self.d), 2))
            * self.E_L(t=0.7 * (27 - self.t + self.d))
        ) / 2.0

    @property
    def I_S(self):
        return 1.0 - 0.23 * self.c - 0.97 * self.m

    @property
    def I_E(self):
        return self.a * self.I_S

    def calc_Gt(
        self,
        t: Optional[ndarray | float | int] = None,
        dt: Optional[float] = None,
        n: int = 1,
    ) -> DataFrame:
        """
        Calculates Gt for given time frame.

        Args:
            t (ndarray or None): Array of time values. Defaults to None.
            dt (float or None): Time delta (only used if `t` is None). Defaults to None.
            n (int): Number of time steps (only used if `t` is None). Defaults to 1.

        Returns:
            DataFrame: A DataFrame containing Gt values for each column index, with time values as the index.
        """
        if isinstance(t, (float, int)):
            t = array([t], dtype=float)
        elif t is None:
            if dt is None:
                dt = np_abs(self.t[-1] - self.t[-2])
            t = mod(
                linspace(
                    # type: ignore
                    self.t[-1] + dt,
                    self.t[-1] + (n - 1) * dt,
                    num=n,
                ),
                24,
            )
        Gt = vectorized_G(
            # type: ignore
            t,
            self.I_E[-1],
            self.tM,
            self.taug,
            self.B,
            self.Cm,
            self.toff,
        )
        df_gt = DataFrame(
            {"Gt{}".format(i): Gt[i] for i in range(Gt.shape[0])}, index=t
        )  # type: ignore
        df_gt["Gt"] = nansum(Gt, axis=0)
        return df_gt

    def update_Gt(self, t=None, dt=None, n=1, keep_tvec_size=True):
        """
        Update the Gt values of the object.

        Parameters:
            t (float or None): The starting time point for the calculation. If None, the calculation starts from the last time point.
            dt (float or None): The time step size for the calculation. If None, the time step size is determined automatically.
            n (int): The number of time steps to calculate.
            keep_tvec_size (bool): Whether to keep the size of the time vector constant. If True, the size of the time vector will be reduced by `n`.

        Returns:
            df_gt (DataFrame): The updated Gt values.

        """
        df_gt = self.calc_Gt(t=t, dt=dt, n=n)
        self.t = append(self.t, df_gt.index.to_numpy(dtype=float))
        if keep_tvec_size is True:
            self.t = self.t[n:]
        return df_gt

    @property
    def G(self):
        """G: get the post-prandial glucose dynamics.

        Returns:
            np.ndarray: Array of Post-prandial glucose dynamics.
        """
        return vectorized_G(self.t, self.I_E, self.tM, self.taug, self.B, self.Cm, self.toff)  # type: ignore

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

    def integrate_signal(
        self,
        signal: Optional[ndarray] = None,
        signal_name: Optional[str] = None,
        t0: int | float = 0,
        t1: int | float = 24,
        M: int = 3,
        t_extra: Optional[Tuple] = None,
        tvec: Optional[ndarray] = None,
    ) -> float:
        """Integrate the signal between the hours given, assuming M discrete events.

        t_extra specifies any additional range of 'accepted hours' as an inclusive tuple [te0, te1],
        to be included in the target time period.
        """
        # trunk-ignore(bandit/B101)
        assert any(
            [(signal is None), (signal_name is None)]
        ), "Must provide exactly one of signal or signal_name"
        # trunk-ignore(bandit/B101)
        assert any(
            [(signal is not None), (signal_name is not None)]
        ), "Must provide exactly one of signal or signal_name"
        if tvec is None:
            tvec = self.t  # type: ignore
        if signal_name is not None:
            signal = getattr(self, signal_name)
        if signal.shape[0] != tvec.size:  # type: ignore
            signal = signal.T  # type: ignore
        period = logical_and((tvec >= t0), (tvec <= t1))  # type: ignore
        if t_extra is not None:
            period = logical_or(
                period, (tvec >= t_extra[0]) & (tvec <= t_extra[1]))
        total = nansum(signal[period]) / (M * (t1 - t0))  # type: ignore
        return total

    def morning(self, signal: Optional[ndarray] = None, signal_name=None):
        """compute the total morning integrated signal."""
        return self.integrate_signal(
            signal=signal, signal_name=signal_name, t0=4, t1=13
        )

    def evening(self, signal: Optional[ndarray] = None, signal_name=None):
        """Compute the total evening integrated signal."""
        return self.integrate_signal(
            signal=signal, signal_name=signal_name, t0=16, t1=24, t_extra=(0, 3)
        )

    @property
    def columns(self) -> Tuple[str]:
        """column names for the DataFrame.

        Examples:
        ---------
            >>> cma = CMASleepWakeModel(N=4)
            >>> cma.columns
            ('t', 'c', 'm', 'a', 'I_S', 'I_E', 'L', 'G')

        Returns:
            tuple[str]: column names for the Dataframe
        """
        return ("t", "c", "m", "a", "I_S", "I_E", "L", "G")  # type: ignore

    @property
    def I_morning(self):
        return self.morning(signal_name="I_S")

    @property
    def I_evening(self):
        return self.evening(signal_name="I_S")

    @property
    def g_morning(self):
        return self.morning(self.g)

    @property
    def g_evening(self):
        return self.evening(self.g)

    @property
    def g_instant(self):
        """vector of instantaneous (overall) glucose."""
        return nansum(self.g, axis=0)

    @property
    def df(self) -> DataFrame:
        return self.run()

    @property
    def dt(self):
        #: TimedeltaIndex (in hours)
        # use 'h' to avoid deprecation warning
        return to_timedelta(self.t, unit="h")

    @property
    def pvec(self):
        """easy access to parameter vector (copy)"""
        return array([self.params[k] for k in self.param_keys])

    def run(self) -> DataFrame:
        """run the model, return the solution as a labeled DataFrame.

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
        df = DataFrame(data, columns=columns, index=self.dt)
        #: record instantaneous glucose
        df["G"] = self.g_instant
        #: record estimated meal times
        ismeal = []
        for tm in self.tM:
            time_since_meal = (df["t"] - tm).abs()
            if len(time_since_meal) == 0:
                continue
            ismeal.append(time_since_meal.idxmin())
        df["is_meal"] = False
        if len(ismeal) > 0:
            df.loc[ismeal, "is_meal"] = True
        return df


def round_to_nearest_integer(number):
    rounded_number = round(number, 2)
    return int(rounded_number)


class CMAUtils:

    @staticmethod
    def get_hour_of_day(
        hour: Tuple[float | int] | float | int,
    ) -> int | str | Tuple[str | int]:
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
            return tuple(map(CMAUtils.get_hour_of_day, hour))  # type: ignore
        if not isinstance(hour, (float, int)):
            raise ValueError("The hour must be a float or integer value.")
        if hour < 0 or hour > 24:
            raise ValueError("The hour must be between 0 and 24.")
        if hour == 0 or hour == 24:
            return "12AM"
        elif hour == 12:
            return "12PM"
        elif hour < 12:
            return f"{int(hour)}AM"
        else:
            return f"{int(hour) - 12}PM"

    @staticmethod
    def label_meals(
        df: DataFrame,
        rounded: Optional[Callable] = round_to_nearest_integer,
        as_str: bool = False,
    ) -> Tuple[str | int | float]:
        """Label the meal times in a CMA model results dataframe.
        Parameters:
            df (DataFrame): The CMA model results dataframe.
            rounded (None or Callable): Function to round the meal times.
            as_str (bool): If True, return the meal times as strings.
        Returns:
            tuple: The meal times as strings or integers.
        Examples:
            >>> df = DataFrame({'t': [0, 8, 16, 24], 'is_meal': [True, True, True, False]})
            >>> CMAUtils.label_meals(df)
            ('12AM', '8AM', '4PM')
            >>> CMAUtils.label_meals(df, as_str=True)
            ('12AM', '8AM', '4PM')
            >>> CMAUtils.label_meals(df, rounded=round_to_nearest_integer)
            (0, 8, 16)
        """
        #: get the meal times
        mealtimes = df.loc[df["is_meal"], "t"]
        if rounded is not None:
            mealtimes = mealtimes.apply(rounded)
        tM = mealtimes.to_numpy(dtype=float).tolist()
        if as_str:
            tM = CMAUtils.get_hour_of_day(tM)
        if not isinstance(tM, tuple):
            if hasattr(tM, "__iter__"):
                tM = tuple(tM)  # type: ignore
            else:
                tM = (tM,)
        return tM  # type: ignore
