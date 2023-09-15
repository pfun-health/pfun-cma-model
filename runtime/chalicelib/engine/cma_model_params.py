from typing import Optional, Sequence
from pydantic import BaseModel, field_serializer
from numpy import ndarray


class CMAModelParams(BaseModel, arbitrary_types_allowed=True):
    """
    Represents the parameters for a CMA model.

    Args:
        t (Optional[type], optional): The value of t. Defaults to None.
        N (int, optional): The value of N. Defaults to 288.
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
    N: int = 288
    d: float = 0.0
    taup: float = 1.0
    taug: float | Sequence[float] | ndarray = 1.0
    B: float = 0.05
    Cm: float = 0.0
    toff: float = 0.0
    tM: Sequence[float] | float = (7.0, 11.0, 17.5)
    seed: Optional[int | float] = None
    eps: float = 1e-18

    @field_serializer('t')
    def serialize_t(self, value, *args):
        if isinstance(value, ndarray):
            return value.tolist()
        return value
