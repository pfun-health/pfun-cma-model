from typing import Any
from pydantic import WrapSerializer
from pydantic import WithJsonSchema
from pydantic.functional_serializers import model_serializer
from typing import Annotated
from pydantic.functional_serializers import PlainSerializer
import json
from typing import Container
from numbers import Real
from pydantic import create_model
from pydantic import Field
import numpy as np
from dataclasses import dataclass
from pfun_cma_model.misc.types import NumpyArray
import sys
from pathlib import Path
from typing import Optional, Sequence, Dict, Tuple, ClassVar
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
_STEP_DEFAULTS = [0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
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


class BoundedCMAModelParam(Real, BaseModel):
    """Encapsulates a bounded parameter with metadata for the CMA model."""

    """Metadata for the bounded parameter.
    Attributes:
        index (int): Index of the parameter.
        name (str): Name of the parameter.
        value (float): Value of the parameter.
        default (float): Default value of the parameter.
        description (str): Description of the parameter.
        qualitative_descriptor (str): Qualitative description based on the value.
        serr (float): Standardized error value.
        step (float): Step size for the parameter.
        min (float): Minimum bound for the parameter.
        max (float): Maximum bound for the parameter.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: int = -1
    name: str = ''
    value: float | Container[float] = 0.0
    default: float = 0.0
    description: str = ''
    qualitative_descriptor: str = ''
    serr: float = -1
    step: float = 0.01
    min: float = 0.0
    max: float = 1.0

    @field_serializer('index', 'value', 'name', 'description', 'min', 'max', 'step', when_used='json')
    def serialize_value(self, v: float | int, _info):
        return v
    
    def __json__(self):
        return vars(self)

    def __repr__(self):
        return f"{self.__class__.__name__}({vars(self)})"

    def __float__(self):
        return float(self.value)

    def __abs__(self):
        return abs(self.value)

    def __add__(self, other):
        return float(self) + other

    def __ceil__(self):
        import math
        return math.ceil(self.value)

    def __eq__(self, other):
        return float(self) == other

    def __floor__(self):
        import math
        return math.floor(self.value)

    def __floordiv__(self, other):
        return float(self) // other

    def __le__(self, other):
        return float(self) <= other

    def __lt__(self, other):
        return float(self) < other

    def __mod__(self, other):
        return float(self) % other

    def __mul__(self, other):
        return float(self) * other

    def __neg__(self):
        return -float(self)

    def __pos__(self):
        return +float(self)

    def __pow__(self, other):
        return float(self) ** other

    def __radd__(self, other):
        return other + float(self)

    def __rfloordiv__(self, other):
        return other // float(self)

    def __rmod__(self, other):
        return other % float(self)

    def __rmul__(self, other):
        return other * float(self)

    def __round__(self, ndigits=None):
        return round(self.value, ndigits) if ndigits is not None else round(self.value)

    def __rpow__(self, other):
        return other ** float(self)

    def __rtruediv__(self, other):
        return other / float(self)

    def __truediv__(self, other):
        return float(self) / other

    def __trunc__(self):
        import math
        return math.trunc(self.value)


class BaseModelParams(BaseModel):
    """
    Base class for model parameters, providing common functionality.
    This class is not intended to be instantiated directly.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
        raise KeyError(f"'{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value):
        """
        Set a parameter value by key, including bounded parameters.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        elif hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' not found in {self.__class__.__name__}")

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
                raise KeyError(
                    f"'{key}' not found in {self.__class__.__name__}")
        return self


def generate_bounded_params_fields() -> dict[str, Field]:
    """
    Generate a set of fields for the bounded parameters 
    """
    fields = {}
    for ix, name in enumerate(_BOUNDED_PARAM_KEYS_DEFAULTS):
        new_field = BoundedCMAModelParam(
            value=_MID_DEFAULTS[ix],
            index=ix,
            name=name,
            default=_MID_DEFAULTS[ix],
            description=_BOUNDED_PARAM_DESCRIPTIONS[ix],
            min=_LB_DEFAULTS[ix],
            max=_UB_DEFAULTS[ix],
            step=_STEP_DEFAULTS[ix]
        )
        Field()
        fields.update({name: new_field})
    return fields


def serialize_param(value: BoundedCMAModelParam, handler, info) -> str:
    return value.model_dump_json()


ParamFields = generate_bounded_params_fields()
BoundedCMAModelParamsBase = create_model(
    'BoundedCMAModelParamsBase',
    __annotations__={k: Annotated[dict, WrapSerializer(
        serialize_param)] for k in ParamFields},
    __config__=ConfigDict(arbitrary_types_allowed=True),
    **ParamFields
)


class BoundedCMAModelParams(BoundedCMAModelParamsBase, BaseModelParams, BaseModel):
    """
    Encapsulates bounded parameters and their metadata for the CMA model.
    """
    # Metadata (ClassVar for constants, private for instance-specific)
    lb: ClassVar[Sequence[float]] = _LB_DEFAULTS
    ub: ClassVar[Sequence[float]] = _UB_DEFAULTS
    step: ClassVar[Sequence[float]] = _STEP_DEFAULTS
    bounded_param_keys: ClassVar[Tuple[str, ...]
                                 ] = _BOUNDED_PARAM_KEYS_DEFAULTS
    midbound: ClassVar[Sequence[float]] = _MID_DEFAULTS
    bounded_param_descriptions: ClassVar[Tuple[str, ...]
                                         ] = _BOUNDED_PARAM_DESCRIPTIONS
    bounds: ClassVar[BoundsType] = _DEFAULT_BOUNDS

    @model_serializer()
    def serialize_model(self):
        pass  # TODO

    @field_serializer('taug', check_fields=False)
    def serialize_ndarrays(self, value, *args):
        if isinstance(value, ndarray):
            return value.tolist()
        return value

    @property
    def bounded_params_dict(self) -> Dict[str, float]:
        return {key: getattr(self, key).__json__() for key in self.bounded_param_keys}

    def get_bounded_param(self, key: str) -> BoundedCMAModelParam:
        """
        Get a bounded parameter by key.
        Returns a BoundedCMAModelParam instance with metadata.
        """
        if key not in self.bounded_param_keys:
            raise KeyError(f"'{key}' is not a bounded parameter.")
        value = getattr(self, key)
        ix = self.bounded_param_keys.index(key)
        return BoundedCMAModelParam(
            name=key,
            value=value,
            description=self.bounded_param_descriptions[ix],
            step=self.step[ix],
            min=self.bounds.lb[ix],
            max=self.bounds.ub[ix]
        )

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


class CMAModelParams(BaseModelParams, BaseModel):
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
