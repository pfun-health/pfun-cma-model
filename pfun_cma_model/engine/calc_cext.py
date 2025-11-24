# engine/calc_cext.py
"""
NumPy-friendly wrapper around the C extension pfun_cma_engine.
"""
from __future__ import annotations
import numpy as np
from typing import Sequence, Optional
# import the compiled extension module
try:
    import pfun_cma_engine
except ImportError as e:
    raise ImportError("pfun_cma_engine extension not found — build the extension first.") from e

def calc_model(
    t: Sequence[float],
    d: float = 0.0,
    taup: float = 1.0,
    taug: float = 1.0,
    B: float = 0.05,
    Cm: float = 0.0,
    toff: float = 0.0,
    tM: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    eps: float = 1e-18,
) -> np.ndarray:
    """
    Wrapper -> returns numpy array of model values.

    Parameters mirror the C function:
    - t: sequence of times in hours (1D)
    - tM: sequence of meal times (optional)
    """
    # convert to lists (the C extension accepts Python sequences)
    t_list = list(map(float, np.asarray(t).ravel()))
    tM_list = None if tM is None else list(map(float, np.asarray(tM).ravel()))

    if seed is None:
        seed_arg = None
    else:
        seed_arg = int(seed)

    # call the extension
    result_list = pfun_cma_engine.calc_model(
        t_list, float(d), float(taup), float(taug),
        float(B), float(Cm), float(toff), tM_list, seed_arg, float(eps)
    )
    # convert to numpy array
    return np.asarray(result_list, dtype=float)
