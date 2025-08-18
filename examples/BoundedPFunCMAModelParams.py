from pfun_cma_model.engine.cma_model_params import *


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
        ge (float): Minimum bound for the parameter (greater-than-or-equal-to).
        le (float): Maximum bound for the parameter (less-than-or-equal-to).
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
    ge: float = 0.0
    le: float = 1.0

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



if __name__ == '__main__':
    bounded_params = BoundedCMAModelParam()
