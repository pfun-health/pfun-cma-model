import numpy as np


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

    def __init__(self, lb=-np.inf, ub=np.inf, keep_feasible=False):
        self.lb = np.atleast_1d(lb)
        self.ub = np.atleast_1d(ub)
        self.keep_feasible = np.atleast_1d(keep_feasible).astype(bool)
        self._input_validation()

    def __repr__(self):
        start = f"{type(self).__name__}({self.lb!r}, {self.ub!r}"
        if np.any(self.keep_feasible):
            end = f", keep_feasible={self.keep_feasible!r})"
        else:
            end = ")"
        return start + end

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