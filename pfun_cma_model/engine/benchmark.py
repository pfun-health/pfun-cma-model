import time
import numpy as np
from pfun_cma_model.engine.cma import CMASleepWakeModel
from pfun_cma_model.engine.c_wrapper import run_cma_engine_c

def benchmark():
    # Setup common parameters
    N = 1024
    t = np.linspace(0, 24, N)
    d, taup, taug, B, Cm, toff = 0.0, 1.0, 1.0, 0.05, 0.0, 0.0
    tM = np.array([7.0, 11.0, 17.5])
    
    print(f"Benchmarking CMA Engine: N={N}, Iterations=100\n")

    # --- Python Benchmark ---
    model = CMASleepWakeModel(N=N, tM=tM)
    model.update(d=d, taup=taup, taug=taug, B=B, Cm=Cm, toff=toff)
    
    start_py = time.perf_counter()
    for _ in range(100):
        # In the real fit loop, the update is called repeatedly
        model.update(d=d, taup=taup, taug=taug, B=B, Cm=Cm, toff=toff)
        _ = model.g_instant
    end_py = time.perf_counter()
    py_time = end_py - start_py
    print(f"Python Engine: {py_time:.4f}s total ({py_time/100:.6f}s per call)")

    # --- C Benchmark ---
    start_c = time.perf_counter()
    for _ in range(100):
        res = run_cma_engine_c(
            t=t, d=d, taup=taup, taug_val=taug, 
            B=B, Cm=Cm, toff=toff, tM=tM
        )
        _ = res["G"]
    end_c = time.perf_counter()
    c_time = end_c - start_c
    print(f"C Engine:      {c_time:.4f}s total ({c_time/100:.6f}s per call)")

    # --- Verification ---
    py_G = model.g_instant
    c_G = res["G"]
    diff = np.abs(py_G - c_G).max()
    print(f"\nMax Difference: {diff:.2e}")
    
    speedup = py_time / c_time
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark()
