"""Numba-optimized calculations.
"""
import numpy as np
import pandas as pd


def expit_pfun(x):
    return 1.0 / (1.0 + np.exp(-2.0 * x))


def calc_vdep_current(v, v1, v2, A=1.0, B=1.0):
    return A * expit_pfun(B * (v - v1) / v2)


def E_norm(x):
    y = 2.0 * (expit_pfun(2.0 * x) - 0.5)
    return y


def _normalize(x, a, b):
    """normalize a flattened float array between a and b
    """
    xmin, xmax = x.min(), x.max()
    return a + (b - a) * (x - xmin) / (xmax - xmin)


def normalize(x, a: float = 0.0, b: float = 1.0):
    """normalize a flat 1-d np.ndarray[float] between a and b
    """
    if isinstance(x, pd.Series):
        x = x.to_numpy(dtype=float, na_value=np.nan)
    assert x.ndim < 2
    x = np.array(x, dtype=float).flatten()
    return _normalize(x, a, b)
