"""Numba-optimized calculations.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import importlib
try:
    from chalicelib.decorators import check_is_numpy
except ModuleNotFoundError:
    root_path = str(Path(__file__).parents[1])
    mod_path = str(Path(__file__).parent)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    if mod_path not in sys.path:
        sys.path.insert(0, mod_path)
    check_is_numpy = importlib.import_module(
        ".decorators", package="chalicelib").check_is_numpy


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
        return 1.0 / (1.0 + np.exp(-2*x))
    numer = (8.95 * np.power((G - g_s), 3) +
             np.power((G - g0), 2) - np.power((G - g1), 2))
    return 2.0 * E(1e-4 * numer / (g1 - g0))