import json
from pfun_cma_model.misc.types import NumpyArray
from argparse import Namespace
from typing import Optional, Sequence, Dict, Tuple, ClassVar, Literal
from pydantic import BaseModel, field_serializer, ConfigDict, Field  # type: ignore
from numpy import ndarray, array, linspace
from tabulate import tabulate  # type: ignore
import pfun_path_helper  # type: ignore
from typing import Annotated, Iterable, Any
import pfun_cma_model.engine.bounds as bounds

# import custom ndarray schema

__all__ = [
    'CMAModelParams',
    'CMABoundedParams',
    'QualsMap'
]

# import custom bounds types

Bounds = bounds.Bounds  # necessary for typing (linter)
# BoundsType = type[bounds.BoundsType]  # Removed because bounds.BoundsType is not defined

_LB_DEFAULTS = (-12.0, 0.5, 0.1, 0.0, 0.0, -3.0)
_MID_DEFAULTS = (0.0, 1.0, 1.0, 0.05, 0.0, 0.0)
_UB_DEFAULTS = (14.0, 3.0, 3.0, 1.0, 2.0, 3.0)
_STEP_DEFAULTS = (0.05, 0.01, 0.01, 0.01, 0.01, 0.01)
_BOUNDED_PARAM_KEYS_DEFAULTS = (
    'd', 'taup', 'taug', 'B', 'Cm', 'toff'
)
_EPS = 0.1 + 1e-8
_BOUNDED_PARAM_DESCRIPTIONS = (
    'Time zone offset (hours)',
    'Photoperiod length (hours)',
    'Glucose response time constant',
    'Glucose Bias constant (baseline glucose level)',
    'Cortisol temporal sensitivity coefficient',
    'Solar noon offset (effects of latitude)'
)


class QualsMap:
    def __init__(self, serr):
        self.serr = serr

    @property
    def qualitative_descriptor(self):
        """Generate a qualtitative description, use docstrings for matching conditions."""
        desc = ''
        for attr in ('very', 'low', 'normal', 'high'):
            if getattr(self, attr):
                desc += f'{attr} '
        return desc.strip().title()

    @property
    def low(self):
        """Low"""
        return self.serr <= -_EPS

    @property
    def high(self):
        """High"""
        return self.serr >= _EPS

    @property
    def normal(self):
        """Normal"""
        return self.serr >= -_EPS and self.serr <= _EPS

    @property
    def very(self):
        """Very"""
        return abs(self.serr) >= 0.23


_DEFAULT_BOUNDS = Bounds(
    lb=_LB_DEFAULTS,
    ub=_UB_DEFAULTS,
    keep_feasible=Bounds.True_
)


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
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def bounded_param_keys(self):
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
        Cm (float, optional): Cortisol temporal sensitivity coefficient. Defaults to 0.0.
        toff (float, optional): Solar noon offset (latitude). Defaults to 0.0.
        tM (Tuple[float, float, float], optional): Meal times (hours). Defaults to (7.0, 11.0, 17.5).
        seed (Optional[int], optional): Random seed. Set to an integer to enable random noise via parameter 'eps'. Defaults to None.
        eps (float, optional): Random noise scale ("epsilon"). Defaults to 1e-18.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    tM: Any | Annotated[ndarray, NumpyArray] | float = array([7.0, 11.0, 17.5])
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
    lb: ClassVar[float | Sequence[float]] = _LB_DEFAULTS
    """
    Lower bounds for bounded parameters. Defaults to _LB_DEFAULTS.
    """
    ub: ClassVar[float | Sequence[float]] = _UB_DEFAULTS
    """
    Upper bounds for bounded parameters. Defaults to _UB_DEFAULTS.
    """
    bounded_param_keys: ClassVar[Iterable[str] | Sequence[str]
                                 | Tuple[str]] = _BOUNDED_PARAM_KEYS_DEFAULTS
    """
    Keys for bounded parameters. Defaults to _BOUNDED_PARAM_KEYS_DEFAULTS.
    """
    midbound: ClassVar[Sequence[float]] = _MID_DEFAULTS
    """
    Midpoint values for bounded parameters. Defaults to _MID_DEFAULTS.
    """
    bounded_param_descriptions: ClassVar[Sequence[str] | Tuple[str]] = \
        _BOUNDED_PARAM_DESCRIPTIONS
    """
    Descriptions for bounded parameters. Defaults to _BOUNDED_PARAM_DESCRIPTIONS.
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
            setattr(self, key, value)

    @field_serializer('taug', 'tM', check_fields=False)
    def serialize_ndarrays(self, value, *args):
        if isinstance(value, ndarray):
            return value.tolist()
        return value

    @property
    def t(self):
        """t : ndarray
        Time vector (decimal hours). Generated using new_tvector, using N.
        """
        return self.new_tvector(0, 24, self.N)

    def new_tvector(self, t0: int | float, t1: int | float, n: int) -> ndarray:
        """Create a new linear time vector, given initial (t0), final (t1), and number of timepoints (n)"""
        return linspace(t0, t1, num=int(n))

    @field_serializer('t', check_fields=False, when_used='json')
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
            max=self.bounds.ub[ix]
        )

    def calc_serr(self, param_key: str):
        """Calculate the standardized error (serr) for a bounded parameter."""
        x = getattr(self, param_key)
        ix = list(self.bounded_param_keys).index(param_key)
        mid = self.midbound[ix]
        serr = (x - mid) / (self.bounds.ub[ix] - self.bounds.lb[ix])
        return serr

    def generate_qualitative_descriptor(self, param_key: str):
        """Generate a qualitative descriptor for a bounded parameter."""
        return QualsMap(self.calc_serr(param_key)).qualitative_descriptor

    def describe(self, param_key: str):
        """Generate a description for a bounded parameter."""
        ix = list(self.bounded_param_keys).index(param_key)
        description = self.bounded_param_descriptions[ix]
        return description + ' (' + self.generate_qualitative_descriptor(param_key) + ')'

    def generate_markdown_table(self, output_fmt: Literal["json", "html", "md"], included_params: list[str] | None = None) -> str:
        """Generate a markdown table of the bounded parameters."""
        # Generate content for only the included parameters (if included_params is not None)
        included_params: list[str] = included_params or list(self.bounded_param_keys)  # type: ignore
        table = []
        for param_key in included_params:
            table.append([
                param_key,
                'float',
                getattr(self, param_key),
                self.midbound[list(self.bounded_param_keys).index(param_key)],
                self.bounds.lb[list(self.bounded_param_keys).index(param_key)],
                self.bounds.ub[list(self.bounded_param_keys).index(param_key)],
                self.describe(param_key)
            ])  # type: ignore
        match output_fmt:
            case "md":
                return tabulate(table, headers=['Parameter', 'Type', 'Value', 'Default', 'Lower Bound', 'Upper Bound', 'Description'], tablefmt='github')
            case "html":
                return tabulate(table, headers=['Parameter', 'Type', 'Value', 'Default', 'Lower Bound', 'Upper Bound', 'Description'], tablefmt='html')
            case "json":
                return json.dumps({"table": tabulate(table, headers=['Parameter', 'Type', 'Value', 'Default', 'Lower Bound', 'Upper Bound', 'Description'], tablefmt='github')})
