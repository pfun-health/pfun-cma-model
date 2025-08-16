from pfun_cma_model.misc.types import NumpyArray
import sys
from pathlib import Path
from typing import Annotated, Optional, Sequence, Dict, Tuple, ClassVar
from pydantic import BaseModel, field_serializer, ConfigDict
from numpy import ndarray
from tabulate import tabulate
import importlib
from pfun_path_helper import append_path
append_path(Path(__file__).parent.parent.parent)

# import custom ndarray schema

__all__ = [
    'CMAModelParams',
    'QualsMap'
]

# import custom bounds types
bounds = importlib.import_module('.engine.bounds', package='pfun_cma_model')
Bounds = bounds.Bounds  # necessary for typing (linter)
BoundsType = type[bounds.BoundsType]

_LB_DEFAULTS = [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0]
_MID_DEFAULTS = [0.0, 1.0, 1.0, 0.05, 0.0, 0.0]
_UB_DEFAULTS = [14.0, 3.0, 3.0, 1.0, 2.0, 3.0]
_BOUNDED_PARAM_KEYS_DEFAULTS = (
    'd', 'taup', 'taug', 'B', 'Cm', 'toff'
)
_EPS = 0.1 + 1e-8
_BOUNDED_PARAM_DESCRIPTIONS = (
    'Time zone offset (hours)',
    'Photoperiod length',
    'Glucose response time constant',
    'Glucose Bias constant',
    'Cortisol temporal sensitivity coefficient',
    'Solar noon offset (latitude)'
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


class BoundedCMAModelParams(BaseModel):
    """
    Encapsulates bounded parameters and their metadata for the CMA model.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Bounded parameters
    d: float = 0.0
    taup: float = 1.0
    taug: float | NumpyArray = 1.0
    B: float = 0.05
    Cm: float = 0.0
    toff: float = 0.0

    # Metadata (ClassVar for constants, private for instance-specific)
    lb: ClassVar[Sequence[float]] = _LB_DEFAULTS
    ub: ClassVar[Sequence[float]] = _UB_DEFAULTS
    bounded_param_keys: ClassVar[Tuple[str, ...]
                                 ] = _BOUNDED_PARAM_KEYS_DEFAULTS
    midbound: ClassVar[Sequence[float]] = _MID_DEFAULTS
    bounded_param_descriptions: ClassVar[Tuple[str, ...]
                                         ] = _BOUNDED_PARAM_DESCRIPTIONS
    bounds: ClassVar[BoundsType] = _DEFAULT_BOUNDS

    def get(self, key: str, default=None):
        """
        Get a parameter value by key, including bounded parameters.
        """
        if hasattr(self, key):
            return getattr(self, key)
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        """
        Get a parameter value by key, including bounded parameters.
        """
        if hasattr(self, key):
            return getattr(self, key)
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in CMAModelParams")

    def __setitem__(self, key: str, value):
        """
        Set a parameter value by key, including bounded parameters.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        elif hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' not found in CMAModelParams")

    def update(self, **kwargs):
        """
        Update parameters with keyword arguments, including bounded parameters.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"'{key}' not found in CMAModelParams")
        return self

    @field_serializer('taug')
    def serialize_ndarrays(self, value, *args):
        if isinstance(value, ndarray):
            return value.tolist()
        return value

    @property
    def bounded_params_dict(self) -> Dict[str, float]:
        return {key: getattr(self, key) for key in self.bounded_param_keys}

    def calc_serr(self, param_key: str):
        x = getattr(self, param_key)
        ix = list(self.bounded_param_keys).index(param_key)
        mid = self.midbound[ix]
        serr = (x - mid) / (self.bounds.ub[ix] - self.bounds.lb[ix])
        return serr

    def generate_qualitative_descriptor(self, param_key: str):
        return QualsMap(self.calc_serr(param_key)).qualitative_descriptor

    def describe(self, param_key: str):
        ix = list(self.bounded_param_keys).index(param_key)
        description = self.bounded_param_descriptions[ix]
        return description + ' (' + self.generate_qualitative_descriptor(param_key) + ')'

    def generate_markdown_table(self):
        table = []
        for param_key in self.bounded_param_keys:
            table.append([
                param_key,
                'float',
                getattr(self, param_key),
                self.midbound[list(self.bounded_param_keys).index(param_key)],
                self.bounds.lb[list(self.bounded_param_keys).index(param_key)],
                self.bounds.ub[list(self.bounded_param_keys).index(param_key)],
                self.describe(param_key)
            ])
        return tabulate(table, headers=['Parameter', 'Type', 'Value', 'Default', 'Lower Bound', 'Upper Bound', 'Description'])


class CMAModelParams(BaseModel):
    """
    Represents the parameters for a CMA model, including bounded and unbounded parameters.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Unbounded parameters
    t: Optional[float | NumpyArray] = None
    N: int | None = 24
    tM: Sequence[float] | float = (7.0, 11.0, 17.5)
    seed: Optional[int | float] = None
    eps: Optional[float] = 1e-18

    # Bounded parameters (delegated)
    bounded: BoundedCMAModelParams = BoundedCMAModelParams()

    def get(self, key: str, default=None):
        """
        Get a parameter value by key, including bounded parameters.
        """
        if hasattr(self.bounded, key):
            return getattr(self.bounded, key)
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        """
        Get a parameter value by key, including bounded parameters.
        """
        if hasattr(self.bounded, key):
            return getattr(self.bounded, key)
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in CMAModelParams")

    def __setitem__(self, key: str, value):
        """
        Set a parameter value by key, including bounded parameters.
        """
        if hasattr(self.bounded, key):
            setattr(self.bounded, key, value)
        elif hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' not found in CMAModelParams")

    def update(self, **kwargs):
        """
        Update parameters with keyword arguments, including bounded parameters.
        """
        for key, value in kwargs.items():
            if hasattr(self.bounded, key):
                self.bounded[key] = value
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"'{key}' not found in CMAModelParams")
        return self

    def __getattr__(self, name):
        # Forward attribute requests for bounded params to self.bounded
        if name in self.bounded.bounded_param_keys:
            return getattr(self.bounded, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Forward attribute setting for bounded params to self.bounded
        if name in self.bounded.bounded_param_keys:
            setattr(self.bounded, name, value)
        else:
            # Handle unbounded parameters or raise an error
            if hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'")
        return super().__setattr__(name, value)

    @property
    def bounds(self) -> BoundsType:
        return self.bounded.bounds

    @property
    def bounded_params_dict(self) -> Dict[str, float]:
        return self.bounded.bounded_params_dict

    def calc_serr(self, param_key: str):
        return self.bounded.calc_serr(param_key)

    def generate_qualitative_descriptor(self, param_key: str):
        return self.bounded.generate_qualitative_descriptor(param_key)

    def describe(self, param_key: str):
        return self.bounded.describe(param_key)

    def generate_markdown_table(self):
        return self.bounded.generate_markdown_table()
