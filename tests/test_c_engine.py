import ctypes
import os
import numpy as np
from pfun_cma_model.engine.cma import CMASleepWakeModel

# Load the shared library
lib_path = os.path.abspath("pfun_cma_model/engine/libpfun_cma_engine.so")
lib = ctypes.CDLL(lib_path)

# Define argument types for run_cma_model
lib.run_cma_model.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, # t, N
    ctypes.c_double, ctypes.c_double, ctypes.c_double, # d, taup, taug_val
    ctypes.POINTER(ctypes.c_double), # taug_vec
    ctypes.c_double, ctypes.c_double, ctypes.c_double, # B, Cm, toff
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, # tM, n_meals
    ctypes.POINTER(ctypes.c_int), ctypes.c_double, # seed, eps
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), # out_L, out_m
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), # out_c, out_a
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), # out_I_S, out_I_E
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)  # out_G, out_g
]

def test_c_engine():
    # Setup parameters
    N = 24
    t = np.linspace(0, 24, N)
    d = 0.0
    taup = 1.0
    taug_val = 1.0
    B = 0.05
    Cm = 0.0
    toff = 0.0
    tM = np.array([7.0, 11.0, 17.5], dtype=np.float64)
    n_meals = len(tM)
    seed = ctypes.c_int(0)
    eps = 1e-18

    # Allocate output buffers
    out_L = np.zeros(N, dtype=np.float64)
    out_m = np.zeros(N, dtype=np.float64)
    out_c = np.zeros(N, dtype=np.float64)
    out_a = np.zeros(N, dtype=np.float64)
    out_I_S = np.zeros(N, dtype=np.float64)
    out_I_E = np.zeros(N, dtype=np.float64)
    out_G = np.zeros(N, dtype=np.float64)
    out_g = np.zeros(n_meals * N, dtype=np.float64)

    # Run C engine
    lib.run_cma_model(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), N,
        d, taup, taug_val,
        None, # taug_vec
        B, Cm, toff,
        tM.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n_meals,
        ctypes.byref(seed), eps,
        out_L.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_I_S.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_I_E.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_G.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_g.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    # Run Python engine
    model = CMASleepWakeModel(N=N, d=d, taup=taup, taug=taug_val, B=B, Cm=Cm, toff=toff, tM=tM, seed=None, eps=eps)
    py_L = model.L
    py_m = model.m
    py_c = model.c
    py_a = model.a
    py_I_S = model.I_S
    py_I_E = model.I_E
    py_G = model.g_instant

    # Compare results
    print("Comparing C and Python results...")
    np.testing.assert_allclose(out_L, py_L, rtol=1e-5, atol=1e-8, err_msg="L mismatch")
    np.testing.assert_allclose(out_m, py_m, rtol=1e-5, atol=1e-8, err_msg="m mismatch")
    np.testing.assert_allclose(out_c, py_c, rtol=1e-5, atol=1e-8, err_msg="c mismatch")
    np.testing.assert_allclose(out_a, py_a, rtol=1e-5, atol=1e-8, err_msg="a mismatch")
    np.testing.assert_allclose(out_I_S, py_I_S, rtol=1e-5, atol=1e-8, err_msg="I_S mismatch")
    np.testing.assert_allclose(out_I_E, py_I_E, rtol=1e-5, atol=1e-8, err_msg="I_E mismatch")
    np.testing.assert_allclose(out_G, py_G, rtol=1e-5, atol=1e-8, err_msg="G mismatch")
    print("All checks passed!")

if __name__ == "__main__":
    test_c_engine()
