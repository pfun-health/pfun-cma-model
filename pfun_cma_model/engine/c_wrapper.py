import ctypes
import numpy as np
from typing import Tuple, List, Optional

# Load the shared library
try:
    _lib = ctypes.CDLL("./pfun_cma_engine.so")
except OSError:
    # Fallback for absolute path if relative fails
    _lib = ctypes.CDLL("/home/robbiec/Git/pfun-cma-model/pfun_cma_model/engine/pfun_cma_engine.so")

# --- Function Prototypes ---

# double exp_clipped(double x)
_exp_clipped = _lib.exp_clipped
_exp_clipped.argtypes = [ctypes.c_double]
_exp_clipped.restype = ctypes.c_double

# void run_cma_model(
#     const double* t, int N,
#     double d, double taup, double taug_val,
#     const double* taug_vec,
#     double B, double Cm, double toff,
#     const double* tM, int n_meals,
#     int* seed, double eps,
#     double* out_L, double* out_m, double* out_c, double* out_a,
#     double* out_I_S, double* out_I_E, double* out_G, double* out_g
# )
_run_cma_model = _lib.run_cma_model
_run_cma_model.argtypes = [
    ctypes.POINTER(ctypes.c_double), # t
    ctypes.c_int,                   # N
    ctypes.c_double,                # d
    ctypes.c_double,                # taup
    ctypes.c_double,                # taug_val
    ctypes.POINTER(ctypes.c_double), # taug_vec
    ctypes.c_double,                # B
    ctypes.c_double,                # Cm
    ctypes.c_double,                # toff
    ctypes.POINTER(ctypes.c_double), # tM
    ctypes.c_int,                   # n_meals
    ctypes.POINTER(ctypes.c_int),    # seed
    ctypes.c_double,                # eps
    ctypes.POINTER(ctypes.c_double), # out_L
    ctypes.POINTER(ctypes.c_double), # out_m
    ctypes.POINTER(ctypes.c_double), # out_c
    ctypes.POINTER(ctypes.c_double), # out_a
    ctypes.POINTER(ctypes.c_double), # out_I_S
    ctypes.POINTER(ctypes.c_double), # out_I_E
    ctypes.POINTER(ctypes.c_double), # out_G
    ctypes.POINTER(ctypes.c_double), # out_g
]
_run_cma_model.restype = None

def run_cma_engine_c(
    t: np.ndarray,
    d: float,
    taup: float,
    taug_val: float,
    taug_vec: Optional[np.ndarray] = None,
    B: float = 0.05,
    Cm: float = 0.0,
    toff: float = 0.0,
    tM: np.ndarray = np.array([7.0, 11.0, 17.5]),
    seed: Optional[int] = None,
    eps: float = 1e-18,
):
    N = len(t)
    n_meals = len(tM)
    
    # Prepare input arrays
    t_ptr = t.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    tM_ptr = tM.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    taug_ptr = None
    if taug_vec is not None:
        taug_ptr = taug_vec.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    seed_val = ctypes.c_int(seed) if seed is not None else None
    seed_ptr = ctypes.pointer(seed_val) if seed_val is not None else None
    
    # Pre-allocate output buffers
    out_L = np.zeros(N, dtype=np.float64)
    out_m = np.zeros(N, dtype=np.float64)
    out_c = np.zeros(N, dtype=np.float64)
    out_a = np.zeros(N, dtype=np.float64)
    out_I_S = np.zeros(N, dtype=np.float64)
    out_I_E = np.zeros(N, dtype=np.float64)
    out_G = np.zeros(N, dtype=np.float64)
    out_g = np.zeros(n_meals * N, dtype=np.float64)
    
    # Map output pointers
    L_ptr = out_L.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    m_ptr = out_m.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_ptr = out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    a_ptr = out_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    IS_ptr = out_I_S.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    IE_ptr = out_I_E.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    G_ptr = out_G.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    g_ptr = out_g.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    # Execute C function
    _run_cma_model(
        t_ptr, N, d, taup, taug_val, taug_ptr, 
        B, Cm, toff, tM_ptr, n_meals, seed_ptr, eps,
        L_ptr, m_ptr, c_ptr, a_ptr, IS_ptr, IE_ptr, G_ptr, g_ptr
    )
    
    return {
        "G": out_G,
        "g": out_g.reshape((n_meals, N)),
        "I_E": out_I_E,
        "L": out_L,
        "m": out_m
    }
