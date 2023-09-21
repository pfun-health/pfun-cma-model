import sys
from pathlib import Path
from typing import Annotated, Optional, Sequence, Dict
from pydantic import BaseModel, field_serializer
from numpy import ndarray
import importlib
from pfun_path_helper import append_path
append_path(Path(__file__).parent.parent.parent)
# from chalicelib.engine.bounds import Bounds
bounds = importlib.import_module('.engine.bounds', package='chalicelib')
Bounds = bounds.Bounds
BoundsType = bounds.BoundsType

_LB_DEFAULTS = (-12.0, 0.5, 0.1, 0.0, 0.0, -3.0)
_UB_DEFAULTS = (14.0, 3.0, 3.0, 1.0, 2.0, 3.0)
_BOUNDED_PARAM_KEYS_DEFAULTS = ('d', 'taup', 'taug', 'B', 'Cm', 'toff')
_DEFAULT_BOUNDS = Bounds(
    lb=_LB_DEFAULTS,
    ub=_UB_DEFAULTS,
    keep_feasible=Bounds.True_
)


class CMAModelParams(BaseModel, arbitrary_types_allowed=True):
    """
    Represents the parameters for a CMA model.

    Args:
        t (Optional[type], optional): The value of t. Defaults to None.
        N (int, optional): The value of N. Defaults to 24.
        d (float, optional): The value of d. Defaults to 0.0.
        taup (float, optional): The value of taup. Defaults to 1.0.
        taug (float, optional): The value of taug. Defaults to 1.0.
        B (float, optional): The value of B. Defaults to 0.05.
        Cm (float, optional): The value of Cm. Defaults to 0.0.
        toff (float, optional): The value of toff. Defaults to 0.0.
        tM (Tuple[float, float, float], optional): The value of tM. Defaults to (7.0, 11.0, 17.5).
        seed (Optional[int], optional): The value of seed. Defaults to None.
        eps (float, optional): The value of eps. Defaults to 1e-18.
    """
    t: Optional[float | Sequence[float] | ndarray] = None
    N: int | None = 24
    d: float = 0.0
    taup: float = 1.0
    taug: float | Sequence[float] | ndarray = 1.0
    B: float = 0.05
    Cm: float = 0.0
    toff: float = 0.0
    tM: Sequence[float] | float = (7.0, 11.0, 17.5)
    seed: Optional[int | float] = None
    eps: Optional[float] = 1e-18
    lb: Optional[float | Sequence[float]] = _LB_DEFAULTS
    ub: Optional[float | Sequence[float]] = _UB_DEFAULTS
    bounded_param_keys: Optional[Sequence[str]] = _BOUNDED_PARAM_KEYS_DEFAULTS
    bounds: Optional[Annotated[Dict, BoundsType()]] = _DEFAULT_BOUNDS

    @field_serializer('bounds')
    def serialize_bounds(self, value: Bounds, *args):
        return value.json()

    @field_serializer('t', 'taug', 'tM', 'lb', 'ub')
    def serialize_ndarrays(self, value, *args):
        if isinstance(value, ndarray):
            return value.tolist()
        return value
