"""Numba-optimized calculations.
"""
from numpy import exp as np_exp
from numpy import (
    clip,
    nan,
    power,
    array,
    ndarray,
    log,
    power,
    cos,
    pi,
    piecewise,
    atleast_1d,
    zeros,
)
from pandas import Series
from pathlib import Path
import sys
import importlib

try:
    from pfun_cma_model.misc.decorators import check_is_numpy
except ModuleNotFoundError:
    root_path = str(Path(__file__).parents[1])
    mod_path = str(Path(__file__).parent)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    if mod_path not in sys.path:
        sys.path.insert(0, mod_path)
    check_is_numpy = importlib.import_module(
        ".misc.decorators", package="pfun_cma_model").check_is_numpy


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
    numer = 8.95 * power((G - g_s), 3) + power((G - g0), 2) - power((G - g1), 2)
    return 2.0 * E(1e-4 * numer / (g1 - g0))


def Light(x):
    """
    Calculates the light intensity based on the input value.

    Parameters:
        x (float): The input value.

    Returns:
        float: The calculated light intensity.
    """
    return 2.0 / (1.0 + exp(2.0 * power(x, 2)))


def E(x):
    """
    Compute the exponential function for the given input.

    Parameters:
        x (float): The input value for the exponential function.

    Returns:
        float: The computed exponential value.
    """
    return 1.0 / (1.0 + exp(-2.0 * x))


def meal_distr(Cm, t, toff):
    """Meal distribution function.

    Parameters
    ----------
    Cm : float
        Cortisol temporal sensitivity coefficient (u/h).
    t : array_like
        Time (hours).
    toff : float
        Meal-relative time offset (hours).

    Returns
    -------
    array_like
        Meal distribution function.
    """
    return power(cos(2 * pi * Cm * (t + toff) / 24), 2)


@check_is_numpy
def K(x: ndarray):
    """
    Defines the glucose response function.
    Apply a piecewise function to the input array `x`.

    Parameters:
        x (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: The result of applying the piecewise function to `x`.
    """
    return piecewise(x, [x > 0.0, x <= 0.0], [
        lambda x_: exp(-power(log(2.0 * x_), 2)), 0.0])


def vectorized_G(t: ndarray | float, I_E: ndarray | float,
                 tm: ndarray | float, taug: ndarray | float,
                 B: float, Cm: float, toff: float):
    """Vectorized version of G(t, I_E, tm, taug, B, Cm, toff).

    Parameters
    ----------
    t : array_like | float
        Time vector (hours).
    I_E : float
        Extracellular insulin (u*mg/mL).
    tm : array_like
        Meal times (hours).
    taug : array_like
        Meal duration (hours).
    B : float
        Bias constant.
    Cm : float
        Cortisol temporal sensitivity coefficient (u/h).
    toff : float
        Meal-relative time offset (hours).

    Returns
    -------
    array_like
        G(t, I_E, tm, taug, B, Cm, toff).
    """
    tm = atleast_1d(tm)
    t = atleast_1d(t)
    taug = atleast_1d(taug)

    def Gtmp(tm_: float | ndarray, taug_: float | ndarray):
        k_G = K((t - atleast_1d(tm_)) / power(atleast_1d(taug_), 2))
        return 1.3 * k_G / (1.0 + I_E)

    m = len(tm)
    n = len(t)
    j = 0
    out = zeros((m, n), dtype=float)
    while j < m:
        gtmp = Gtmp(tm[j], taug[j])
        out[j, :] = gtmp
        j = j + 1
    out = out + B * (1.0 + meal_distr(Cm, t, toff))  # ! apply bias constant.
    return out
