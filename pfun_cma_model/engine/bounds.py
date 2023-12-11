from typing import Sequence, Dict, Any, Type
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler
import numpy as np
from pfun_cma_model.misc.errors import BoundsTypeError


#: Aliases for numpy bool types (necessary for type checking).
True_, False_ = np.bool_(True), np.bool_(False)
Bool_ = np.bool_


_BOUNDS_SCHEMA = core_schema.typed_dict_schema({
    "lb": core_schema.typed_dict_field(core_schema.list_schema(core_schema.float_schema())),
    "ub": core_schema.typed_dict_field(core_schema.list_schema(core_schema.float_schema())),
    "keep_feasible": core_schema.typed_dict_field(core_schema.list_schema(core_schema.bool_schema()))
})


class BoundsType:

    def __get_pydantic_core_schema__(
        self,
        source: Type[Any],
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return _BOUNDS_SCHEMA


class Bounds:
    """Bounds constraint on the variables.

    The constraint has the general inequality form::

        lb <= x <= ub

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    lb, ub : dense array_like, optional
        Lower and upper bounds on independent variables. `lb`, `ub`, and
        `keep_feasible` must be the same shape or broadcastable.
        Set components of `lb` and `ub` equal
        to fix a variable. Use ``np.inf`` with an appropriate sign to disable
        bounds on all or some variables. Note that you can mix constraints of
        different types: interval, one-sided or equality, by setting different
        components of `lb` and `ub` as necessary. Defaults to ``lb = -np.inf``
        and ``ub = np.inf`` (no bounds).
    keep_feasible : dense array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. Must be broadcastable with `lb` and `ub`.
        Default is False. Has no effect for equality constraints.
    """

    #: ! important: keep_feasible must have integer dtype (not bool)
    #: ...defined within the Bounds namespace for easy access.
    True_ = True_
    False_ = False_
    Bool_ = bool_ = Bool_

    def _input_validation(self):
        try:
            res = np.broadcast_arrays(self.lb, self.ub, self.keep_feasible)
            self.lb, self.ub, self.keep_feasible = res
        except ValueError as exc:
            message = "`lb`, `ub`, and `keep_feasible` must be broadcastable."
            raise ValueError(message) from exc

    def __json__(self):
        """json serialization."""
        return {
            "lb": self.lb.tolist(),
            "ub": self.ub.tolist(),
            "keep_feasible": self.keep_feasible.tolist()
        }

    def json(self):
        """json serialization."""
        return self.__json__()

    @property
    def array(self):
        """return the bounds as an array with columns [lower, upper, keep_feasible]."""
        return self._array

    @array.setter
    def array(self, arr):
        self._array = arr

    def __iter__(self):
        """get an iterator over the bounds array."""
        return iter(self.array)

    def __getitem__(self, index):
        """get the bounds information at the specified index."""
        return self.array[index]

    @property
    def lb(self):
        return self.array[:, 0]

    @lb.setter
    def lb(self, value):
        self.array[:, 0] = value

    @property
    def ub(self):
        return self.array[:, 1]

    @ub.setter
    def ub(self, value):
        self.array[:, 1] = value

    @property
    def keep_feasible(self):
        return self.array[:, 2]

    @keep_feasible.setter
    def keep_feasible(self, value):
        self.array[:, 2] = value

    def __setitem__(self, index: int | slice | Sequence[int | Any], value):
        """set the bounds information at the specified index."""
        if isinstance(index, slice):
            self._array[:, index] = np.asarray(value, dtype=object)[:]
            self.lb, self.ub, self.keep_feasible = list(zip(*self._array))
            self.lb = np.asarray(self.lb, dtype=np.float64)
            self.ub = np.asarray(self.ub, dtype=np.float64)
            #: ! important: keep_feasible must have np.bool_ or integer dtype (not bool)
            #: reference: https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool_
            self.keep_feasible = np.asarray(self.keep_feasible, dtype=np.bool_)
        elif isinstance(index, int):
            self.lb[index] = np.asarray(value[0])
            self.ub[index] = np.asarray(value[1])
            self.keep_feasible[index] = np.asarray(value[2])
        else:
            raise TypeError("Bounds index must be an integer or slice.")

    def __len__(self):
        return len(self.lb)

    def __eq__(self, __value: object) -> np.bool_:
        nb = Bounds(__value)
        return np.all(np.allclose(self.array, nb.array))

    @classmethod
    def _assemble_array(cls, lb, ub, keep_feasible):
        keep_feasible = np.asarray(keep_feasible, dtype=np.bool_)
        if keep_feasible.size < len(lb):
            keep_feasible = np.tile(keep_feasible, len(lb))
        return np.asarray(list(zip(lb, ub, keep_feasible)))

    def __init__(self, *args, lb: float | Sequence[float] = -np.inf,
                 ub: float | Sequence[float] = np.inf,
                 keep_feasible: np.bool_ = True_):
        if len(args) > 0:
            #: handle Bounds positional argument
            if isinstance(args[0], Bounds):
                lb, ub, keep_feasible = args[0].lb, args[0].ub, args[0].keep_feasible
                if len(args) > 1:
                    raise ValueError(
                        "Too many positional arguments. Expected either one (a Bounds) instance, or 0.")
            else:
                #: handle alternative positional arguments
                lb, ub, keep_feasible = args
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        keep_feasible: np.ndarray = np.asarray(keep_feasible, dtype=np.bool_)
        self._array = self._assemble_array(lb, ub, keep_feasible)
        self._input_validation()

    def __repr__(self):
        start = f"{type(self).__name__}({self.lb!r}, {self.ub!r}"
        if np.any(self.keep_feasible):
            end = f", keep_feasible={self.keep_feasible!r})"
        else:
            end = ")"
        return start + end

    def update_values(self, arr: np.ndarray | Dict) -> np.ndarray | Dict[str, float | int]:
        """
        Update the values of the input array so that they stay within the specified limits.

        Args:
            arr (array-like): The input array that needs to be updated.

        Returns:
            array-like: The updated array with values within the specified limits.

        Raises:
            ValueError: If the length of the input array does not match the length of the lower
                        and upper bound arrays.
        """
        keys = None
        if isinstance(arr, dict):
            keys = list(arr.keys())
            arr = np.array(list(arr.values()))
        if len(arr) != len(self.lb) or len(arr) != len(self.ub):
            raise ValueError(
                "Length of input array does not match the length of lower and upper bound arrays.")

        updated_arr = []
        for i, val in enumerate(arr):
            if not isinstance(val, (float, int)):
                raise BoundsTypeError(i)
            if not self.keep_feasible[i]:
                updated_arr.append(val)  # ! make sure to still append
                continue  # ! skip bounds check if keep_feasible is False for this element
            if val < self.lb[i]:
                updated_arr.append(self.lb[i])
            elif val > self.ub[i]:
                updated_arr.append(self.ub[i])
            else:
                updated_arr.append(val)

        updated_arr = np.array(updated_arr, dtype=arr.dtype)
        if keys is not None:
            updated_arr = {k: v for k, v in zip(keys, updated_arr, strict=True)}
        return updated_arr

    def residual(self, x):
        """Calculate the residual (slack) between the input and the bounds

        For a bound constraint of the form::

            lb <= x <= ub

        the lower and upper residuals between `x` and the bounds are values
        ``sl`` and ``sb`` such that::

            lb + sl == x == ub - sb

        When all elements of ``sl`` and ``sb`` are positive, all elements of
        ``x`` lie within the bounds; a negative element in ``sl`` or ``sb``
        indicates that the corresponding element of ``x`` is out of bounds.

        Parameters
        ----------
        x: array_like
            Vector of independent variables

        Returns
        -------
        sl, sb : array-like
            The lower and upper residuals
        """
        return x - self.lb, self.ub - x
