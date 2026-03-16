"""Defines the CMAModelParams class, which encapsulates the parameters for a CMA model,
including bounded parameters with associated metadata and methods for generating qualitative descriptors and
markdown tables.
"""

import json
from argparse import Namespace
from typing import (
    Annotated,
    Any,
    ClassVar,
    Dict,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    NamedTuple,
)

from numpy import array, linspace, ndarray, nan
from pydantic import BaseModel, ConfigDict, Field, field_serializer
from tabulate import tabulate

import pfun_cma_model.engine.bounds as bounds
from pfun_cma_model.misc.types import NumpyArray

# import custom ndarray schema

__all__ = ["CMAModelParams", "CMABoundedParams", "QualsMap"]

# import custom bounds types

Bounds = bounds.Bounds  # necessary for typing (linter)


class BoundedParamDefaults(NamedTuple):
    """Default values for bounded parameters."""

    @property
    def lbs(self):
        """Alias for lower bounds."""
        return self.lb

    @property
    def ubs(self):
        """Alias for upper bounds."""
        return self.ub

    @property
    def steps(self):
        """Alias for step sizes."""
        return self.step

    @property
    def mids(self):
        """Alias for midpoint values."""
        return self.mid

    eps: float = 0.1 + 1e-8
    #: Epsilon value for defining the model time-integration step size

    lb: tuple = (-12.0, 0.5, 0.1, 0.0, 0.0, -3.0)
    #: Lower bounds for bounded parameters (d, taup, taug, B, Cm, toff)

    mid: tuple = (0.0, 1.0, 1.0, 0.05, 0.0, 0.0)
    #: Midpoint values for bounded parameters (d, taup, taug, B, Cm, toff)

    ub: tuple = (14.0, 3.0, 3.0, 1.0, 2.0, 3.0)
    #: Upper bounds for bounded parameters (d, taup, taug, B, Cm, toff)

    step: tuple = tuple((ub_i + lb_i) * 0.0125 for lb_i, ub_i in zip(lb, ub))  # type: ignore
    #: Step sizes for bounded parameters (d, taup, taug, B, Cm, toff)

    keys: tuple = ("d", "taup", "taug", "B", "Cm", "toff")
    #: Keys for bounded parameters (d, taup, taug, B, Cm, toff)

    descriptions: tuple = (
        (
            "Time zone offset; scalar hours[time]; Estimated effects of "
            "photoperiod offset; correlates with peak light exposure time "
            "relative to solar noon"
        ),
        (
            "Photoperiod duration; scalar hours[time]; Estimated number of "
            "hours of light exposure (relative to darkness) in a 24-hour "
            "period; correlates with light exposure duration"
        ),
        (
            "Glucose meal-response time constant; dimensionless[time]; "
            "correlates with the rate of postprandial glucose metabolism; "
            "higher values indicate slower return to baseline glucose levels "
            "after meals, which can mitigate hypoglycemia risk by increasing "
            "the time until glucose levels drop dangerously low"
        ),
        (
            "Glucose baseline constant; dimensionless[Glucose]; correlates "
            "with basal glucose levels; correlates with A1C-- values higher "
            "than 0.05 indicate elevated baseline glucose levels, which can "
            "increase hyperglycemia risk"
        ),
        (
            "Cortisol sensitivity coefficient; dimensionless[Cortisol]; "
            "correlates with the influence of cortisol on glucose variability; "
            "higher values indicate greater cortisol sensitivity, which can "
            "increase glucose variability, thereby increasing "
            "hyperglycemia/hypoglycemia risk"
        ),
        (
            "Solar-noon offset; hours[time]; correlates with the timing of "
            "solar noon relative to the individual's circadian phase; can "
            "reflect chronotype and influence the alignment of circadian "
            "rhythms with the external light-dark cycle, which can impact "
            "glucose metabolism and overall metabolic health"
        ),
    )
    #: Descriptions for bounded parameters (d, taup, taug, B, Cm, toff)


_DEFAULTS = BoundedParamDefaults()

_LB_DEFAULTS = _DEFAULTS.lb
_MID_DEFAULTS = _DEFAULTS.mid
_UB_DEFAULTS = _DEFAULTS.ub
_STEP_DEFAULTS = _DEFAULTS.step
_BOUNDED_PARAM_KEYS_DEFAULTS = _DEFAULTS.keys
_BOUNDED_PARAM_DESCRIPTIONS = _DEFAULTS.descriptions
_EPS = _DEFAULTS.eps


class QualsMap:
    """Maps standardized error (serr) values to qualitative descriptors for bounded parameters."""

    def __init__(self, serr):
        self.serr = serr

    @property
    def qualitative_descriptor(self):
        """Generate a qualtitative description, use docstrings for matching conditions."""
        desc = ""
        for attr in ("very", "low", "normal", "high"):
            if getattr(self, attr):
                desc += f"{attr} "
        return desc.strip().title()

    @property
    def low(self):
        """Return True if serr indicates a low value."""
        return self.serr <= -_EPS

    @property
    def high(self):
        """Return True if serr indicates a high value."""
        return self.serr >= _EPS

    @property
    def normal(self):
        """Return True if serr indicates a normal value."""
        return self.serr >= -_EPS and self.serr <= _EPS

    @property
    def very(self):
        """Return True if serr indicates a very high or very low value."""
        return abs(self.serr) >= 0.23


_DEFAULT_BOUNDS = Bounds(lb=_LB_DEFAULTS, ub=_UB_DEFAULTS, keep_feasible=Bounds.True_)


class CMAModelParam(BaseModel):
    """Defines a single CMA Model Parameter."""

    name: str
    value: float | int
    serr: float
    description: str


class BoundedCMAModelParam(CMAModelParam):
    """Defines a single *bounded* CMA Model Parameter."""

    lb: int | float = Field(default=nan)
    ub: int | float = Field(default=nan)
    step: int | float = Field(default=nan)

    @property
    def bounds(self):
        return Bounds(lb=[self.lb], ub=[self.ub], keep_feasible=[True])


class CMABoundedParams(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # set default values for bounded parameters
        for key, default_value in zip(self.bounded_param_keys, _MID_DEFAULTS):
            setattr(self, key, default_value)
        # set to any explicitly passed values
        for key in self.bounded_param_keys:
            setattr(self, key, kwargs.get(key, getattr(self, key, None)))

    def __getitem__(self, key):
        return getattr(self, key)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif hasattr(self.__dict__, name):
            return getattr(self.__dict__, name)
        # If the attribute is not found in __dict__, try to get it from the current instance
        if name in dir(self):
            return self.__dict__[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def bounded_param_keys(self):
        """Get the keys for bounded parameters."""
        return _BOUNDED_PARAM_KEYS_DEFAULTS


class CMAModelParams(BaseModel):
    """
    Represents the parameters for a CMA model.

    Args:
        t (Optional[array_like], optional): Time vector (decimal hours). Defaults to None.
        N (int, optional): Number of time points. Defaults to 24.
        d (float, optional): Time zone offset (hours). Defaults to 0.0.
        taup (float, optional): Circadian-relative photoperiod length. Defaults to 1.0.
        taug (float, optional): Glucose response time constant. Defaults to 1.0.
        B (float, optional): Glucose Bias constant. Defaults to 0.05.
        Cm (float, optional): Cortisol temporal sensitivity coefficient.
            Defaults to 0.0.
        toff (float, optional): Solar noon offset (latitude). Defaults to 0.0.
        tM (Tuple[float, float, float], optional): Meal times (hours).
            Defaults to (7.0, 11.0, 17.5).
        seed (Optional[int], optional): Random seed. Set to an integer to
            enable random noise via parameter 'eps'. Defaults to None.
        eps (float, optional): Random noise scale ("epsilon").
            Defaults to 1e-18.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    """
    Time vector (decimal hours). Optional.
    """
    N: int = 1024
    """
    Number of time points. Defaults to 24.
    """
    d: float = 0.0
    """
    Time zone offset (hours). Defaults to 0.0.
    """
    taup: float = 1.0
    """
    Circadian-relative photoperiod length. Defaults to 1.0.
    """
    taug: float | NumpyArray = 1.0
    """
    Glucose response time constant. Defaults to 1.0.
    """
    B: float = 0.05
    """
    Glucose Bias constant. Defaults to 0.05.
    """
    Cm: float = 0.0
    """
    Cortisol temporal sensitivity coefficient. Defaults to 0.0.
    """
    toff: float = 0.0
    """
    Solar noon offset (latitude). Defaults to 0.0.
    """
    tM: Annotated[ndarray, NumpyArray] | Sequence[float] = array([7.0, 11.0, 17.5])
    """
    Meal times (hours). Defaults to (7.0, 11.0, 17.5).
    """
    seed: Optional[int | float] = None
    """
    Random seed. Set to an integer to enable random noise via parameter 'eps'. Optional.
    """
    eps: Optional[float] = 1e-18
    """
    Random noise scale ("epsilon"). Defaults to 1e-18.
    """
    id_tag: Optional[str] = Field(default=None, exclude=True)
    """
    ID tag for the model, for book-keeping purposes. Optional.
    """
    lb: ClassVar[float | Sequence[float]] = _DEFAULTS.lb
    """
    Lower bounds for bounded parameters. Defaults to _DEFAULTS.lb.
    """
    ub: ClassVar[float | Sequence[float]] = _DEFAULTS.ub
    """
    Upper bounds for bounded parameters. Defaults to _DEFAULTS.ub.
    """
    bounded_param_keys: ClassVar[Iterable[str] | Sequence[str] | Tuple[str]] = _DEFAULTS.keys
    """
    Keys for bounded parameters. Defaults to _DEFAULTS.keys.
    """
    midbound: ClassVar[Sequence[float]] = _DEFAULTS.mid
    """
    Midpoint values for bounded parameters. Defaults to _DEFAULTS.mid.
    """
    bounded_param_descriptions: ClassVar[Sequence[str] | Tuple[str]] = _DEFAULTS.descriptions
    """
    Descriptions for bounded parameters. Defaults to _DEFAULTS.descriptions.
    """
    bounds: ClassVar[Any] = _DEFAULT_BOUNDS
    """
    Bounds object for parameter constraints. Defaults to _DEFAULT_BOUNDS.
    """

    def __getitem__(self, key):
        return getattr(self, key)

    def update(self, **kwargs):
        """Update the model parameters."""
        for key, value in kwargs.items():
            if key in self.bounded_param_keys:
                setattr(self, key, value)
            elif hasattr(self, key):
                setattr(self, key, value)
            elif key.startswith("tM"):
                tM_array = self.tM if len(self.tM) > 0 else []
                tM_array = list(self.tM)
                tM_array.append(float(value))
                setattr(self, "tM", tM_array)  # append new values to tM

    @field_serializer("taug", "tM", check_fields=False)
    def serialize_ndarrays(self, value):
        """Serialize taug and tM as lists for JSON output."""
        if isinstance(value, ndarray):
            return value.tolist()
        return value

    @property
    def t(self) -> ndarray:
        """Time vector (decimal hours). Generated using new_tvector, using N."""
        return self.new_tvector(0, 24, self.N)  # type: ignore

    def new_tvector(self, t0: int | float, t1: int | float, n: int) -> ndarray:
        """Create a new linear time vector, given initial (t0), final (t1), and number of timepoints (n)"""
        return linspace(t0, t1, num=int(n))

    @field_serializer("t", check_fields=False, when_used="json")
    def serialize_t(self, value, *args):
        """Serialize t as list for JSON output."""
        if isinstance(value, ndarray):
            return value.tolist()
        return value

    @property
    def bounded_params_dict(self) -> Dict[str, float]:
        """Get a dictionary of bounded parameters."""
        return {key: getattr(self, key) for key in self.bounded_param_keys}

    @property
    def bounded(self) -> CMABoundedParams:
        """Alias for bounded_params_dict."""
        return CMABoundedParams(**self.bounded_params_dict)

    def get_bounded_param(self, key: str) -> dict[str, Any]:
        """
        Get a bounded parameter by key.
        Returns a BoundedCMAModelParam instance with metadata.
        """
        if key not in self.bounded_param_keys:
            raise KeyError(f"'{key}' is not a bounded parameter.")
        value = getattr(self, key)
        ix = list(self.bounded_param_keys).index(key)
        return dict(
            name=key,
            value=value,
            description=self.bounded_param_descriptions[ix],
            step=_STEP_DEFAULTS[ix],
            min=self.bounds.lb[ix],
            max=self.bounds.ub[ix],
        )

    def calc_serr(self, param_key: str):
        """Calculate the standardized error (serr) for a bounded parameter."""
        x = getattr(self, param_key)
        ix = list(self.bounded_param_keys).index(param_key)
        mid = self.midbound[ix]
        serr = (x - mid) / (self.bounds.ub[ix] - self.bounds.lb[ix])
        return serr

    def serr(self, param_key: str):
        """Alias for calc_serr."""
        return self.calc_serr(param_key)

    def generate_qualitative_descriptor(self, param_key: str):
        """Generate a qualitative descriptor for a bounded parameter."""
        return QualsMap(self.calc_serr(param_key)).qualitative_descriptor

    def describe(self, param_key: str):
        """Generate a description for a bounded parameter."""
        ix = list(self.bounded_param_keys).index(param_key)
        description = self.bounded_param_descriptions[ix]
        return description + " (" + self.generate_qualitative_descriptor(param_key) + ")"

    def generate_markdown_table(
        self,
        output_fmt: Literal["json", "html", "md"],
        included_params: Optional[list[str]] = None,  # type: ignore
    ) -> str:
        """Generate a markdown table of the bounded parameters."""
        # Generate content for only the included parameters (if included_params is not None)
        if included_params is None:
            included_params: list[str] = list(self.bounded_param_keys)  # type: ignore
        else:
            included_params: list[str] = included_params  # type: ignore
        table = []
        for param_key in included_params:  # type: ignore
            table.append(
                [
                    param_key,
                    "float",
                    getattr(self, param_key),
                    self.midbound[list(self.bounded_param_keys).index(param_key)],
                    self.bounds.lb[list(self.bounded_param_keys).index(param_key)],
                    self.bounds.ub[list(self.bounded_param_keys).index(param_key)],
                    self.describe(param_key),
                ]
            )  # type: ignore
        match output_fmt:
            case "md":
                return tabulate(
                    table,
                    headers=[
                        "Parameter",
                        "Type",
                        "Value",
                        "Default",
                        "Lower Bound",
                        "Upper Bound",
                        "Description",
                    ],
                    tablefmt="github",
                )
            case "html":
                return tabulate(
                    table,
                    headers=[
                        "Parameter",
                        "Type",
                        "Value",
                        "Default",
                        "Lower Bound",
                        "Upper Bound",
                        "Description",
                    ],
                    tablefmt="html",
                )
            case "json":
                return json.dumps(
                    {
                        "table": tabulate(
                            table,
                            headers=[
                                "Parameter",
                                "Type",
                                "Value",
                                "Default",
                                "Lower Bound",
                                "Upper Bound",
                                "Description",
                            ],
                            tablefmt="github",
                        )
                    }
                )
