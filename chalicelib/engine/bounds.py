import numpy as np
from typing import Container, Dict


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
    def _input_validation(self):
        try:
            res = np.broadcast_arrays(self.lb, self.ub, self.keep_feasible)
            self.lb, self.ub, self.keep_feasible = res
        except ValueError:
            message = "`lb`, `ub`, and `keep_feasible` must be broadcastable."
            raise ValueError(message)

    def __init__(self, lb: float | Container[float] = -np.inf,
                 ub: float | Container[float] = np.inf, keep_feasible=False):
        self.lb = np.atleast_1d(np.asarray(lb))
        self.ub = np.atleast_1d(np.asarray(ub))
        self.keep_feasible = np.atleast_1d(keep_feasible).astype(bool)
        self._input_validation()

    def __repr__(self):
        start = f"{type(self).__name__}({self.lb!r}, {self.ub!r}"
        if np.any(self.keep_feasible):
            end = f", keep_feasible={self.keep_feasible!r})"
        else:
            end = ")"
        return start + end

    def update_values(self, arr: np.ndarray | Dict):
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
            raise ValueError("Length of input array does not match the length of lower and upper bound arrays.")

        updated_arr = []
        for i, val in enumerate(arr):
            if val < self.lb[i]:
                updated_arr.append(self.lb[i])
            elif val > self.ub[i]:
                updated_arr.append(self.ub[i])
            else:
                updated_arr.append(val)

        updated_arr = np.array(updated_arr, dtype=arr.dtype)
        if keys is not None:
            updated_arr = {k: v for k, v in zip(keys, updated_arr)}
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
