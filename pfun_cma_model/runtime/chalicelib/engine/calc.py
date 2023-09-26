"""Numba-optimized calculations.
"""
from numpy import array, nan, power, clip
from numpy import exp as np_exp
from pandas import Series
from pathlib import Path
import sys
import importlib

try:
    from pfun_cma_model.decorators import check_is_numpy
except ModuleNotFoundError:
    root_path = str(Path(__file__).parents[1])
    mod_path = str(Path(__file__).parent)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    if mod_path not in sys.path:
        sys.path.insert(0, mod_path)
    check_is_numpy = importlib.import_module(
        ".decorators", package="pfun_cma_model").check_is_numpy


def exp(x):
    """
        Calculate the exponential of a number. Clip to avoid overflow.

        Parameters:
        x (float): The input number.

        Returns:
        float: The exponential of the input number.
    """
    x_clipped = clip(x, -709, 709)
    result = np_exp(x_clipped)
    return result


def expit_pfun(x):
    return 1.0 / (1.0 + exp(-2.0 * x))


def calc_vdep_current(v, v1, v2, A=1.0, B=1.0):
    return A * expit_pfun(B * (v - v1) / v2)


def E_norm(x):
    y = 2.0 * (expit_pfun(2.0 * x) - 0.5)
    return y


def _normalize(x, a, b):
    """normalize a flattened float array between a and b"""
    xmin, xmax = x.min(), x.max()
    return a + (b - a) * (x - xmin) / (xmax - xmin)


def normalize(x, a: float = 0.0, b: float = 1.0):
    """normalize a flat 1-d ndarray[float] between a and b"""
    if isinstance(x, Series):
        x = x.to_numpy(dtype=float, na_value=nan)
    assert x.ndim < 2
    x = array(x, dtype=float).flatten()
    return _normalize(x, a, b)


@check_is_numpy
def normalize_glucose(G, g0=70, g1=180, g_s=90):
    """Normalize glucose (mg/dL -> [0.0, 2.0]).

        <0.9: low,
        0.9: normal-low,
        0.9-1.5: normal,
        >1.5: high

    see the graph: https://www.desmos.com/calculator/ii4qrawgjo
    """

    def E(x):
        return 1.0 / (1.0 + exp(-2 * x))

    numer = 8.95 * power((G - g_s), 3) + power((G - g0), 2) - power(
        (G - g1), 2)
    return 2.0 * E(1e-4 * numer / (g1 - g0))
