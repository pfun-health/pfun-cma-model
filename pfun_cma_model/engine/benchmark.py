#!/usr/bin/env python
"""Benchmark: Python CMA engine vs. C extension.

Produces several thousand timepoints and compares wall-clock performance
of the pure-Python CMASleepWakeModel against the compiled C extension
(libpfun_cma_engine.so).

Usage:
    python -m pfun_cma_model.engine.benchmark          # from project root
    python pfun_cma_model/engine/benchmark.py           # direct invocation
"""

import ctypes
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Resolve paths so the benchmark works from any cwd
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]  # pfun-cma-model repo root

# Ensure pfun_cma_model is importable
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pfun_cma_model.engine.cma import CMASleepWakeModel  # noqa: E402

# ---------------------------------------------------------------------------
# Locate and load the C shared library directly (no install required)
# ---------------------------------------------------------------------------
_C_ENGINE_REPO = Path(
    os.environ.get(
        "PFUN_CMA_ENGINE_C",
        Path(__file__).resolve().parents[3] / "pfun-cma-engine-c",
    )
)
_LIB_CANDIDATES = [
    _C_ENGINE_REPO / "pfun_cma_engine" / "libpfun_cma_engine.so",
    _C_ENGINE_REPO / "build" / "libpfun_cma_engine.so",
]

_lib: Optional[ctypes.CDLL] = None
for _candidate in _LIB_CANDIDATES:
    if _candidate.exists():
        _lib = ctypes.CDLL(str(_candidate))
        break

if _lib is None:
    print(
        f"ERROR: Could not find libpfun_cma_engine.so.  "
        f"Searched: {[str(c) for c in _LIB_CANDIDATES]}\n"
        f"Build it first:  cd pfun-cma-engine-c && make",
        file=sys.stderr,
    )
    sys.exit(1)

# ---- C function prototype ------------------------------------------------
_run_cma_model = _lib.run_cma_model
_run_cma_model.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # t
    ctypes.c_int,  # N
    ctypes.c_double,  # d
    ctypes.c_double,  # taup
    ctypes.c_double,  # taug_val
    ctypes.POINTER(ctypes.c_double),  # taug_vec (nullable)
    ctypes.c_double,  # B
    ctypes.c_double,  # Cm
    ctypes.c_double,  # toff
    ctypes.POINTER(ctypes.c_double),  # tM
    ctypes.c_int,  # n_meals
    ctypes.POINTER(ctypes.c_int),  # seed (nullable)
    ctypes.c_double,  # eps
    ctypes.POINTER(ctypes.c_double),  # out_L
    ctypes.POINTER(ctypes.c_double),  # out_m
    ctypes.POINTER(ctypes.c_double),  # out_c
    ctypes.POINTER(ctypes.c_double),  # out_a
    ctypes.POINTER(ctypes.c_double),  # out_I_S
    ctypes.POINTER(ctypes.c_double),  # out_I_E
    ctypes.POINTER(ctypes.c_double),  # out_G
    ctypes.POINTER(ctypes.c_double),  # out_g
]
_run_cma_model.restype = None


def _run_c_engine(t, d, taup, taug_val, B, Cm, toff, tM, eps=1e-18):
    """Thin ctypes wrapper around run_cma_model (no external dependency)."""
    N = len(t)
    n_meals = len(tM)

    t_arr = np.ascontiguousarray(t, dtype=np.float64)
    tM_arr = np.ascontiguousarray(tM, dtype=np.float64)

    out_L = np.zeros(N, dtype=np.float64)
    out_m = np.zeros(N, dtype=np.float64)
    out_c = np.zeros(N, dtype=np.float64)
    out_a = np.zeros(N, dtype=np.float64)
    out_I_S = np.zeros(N, dtype=np.float64)
    out_I_E = np.zeros(N, dtype=np.float64)
    out_G = np.zeros(N, dtype=np.float64)
    out_g = np.zeros(n_meals * N, dtype=np.float64)

    _dptr = ctypes.POINTER(ctypes.c_double)

    _run_cma_model(
        t_arr.ctypes.data_as(_dptr),
        ctypes.c_int(N),
        ctypes.c_double(d),
        ctypes.c_double(taup),
        ctypes.c_double(taug_val),
        None,  # taug_vec
        ctypes.c_double(B),
        ctypes.c_double(Cm),
        ctypes.c_double(toff),
        tM_arr.ctypes.data_as(_dptr),
        ctypes.c_int(n_meals),
        None,  # seed
        ctypes.c_double(eps),
        out_L.ctypes.data_as(_dptr),
        out_m.ctypes.data_as(_dptr),
        out_c.ctypes.data_as(_dptr),
        out_a.ctypes.data_as(_dptr),
        out_I_S.ctypes.data_as(_dptr),
        out_I_E.ctypes.data_as(_dptr),
        out_G.ctypes.data_as(_dptr),
        out_g.ctypes.data_as(_dptr),
    )
    return {
        "G": out_G,
        "g": out_g.reshape((n_meals, N)),
        "I_E": out_I_E,
        "L": out_L,
        "m": out_m,
        "c": out_c,
        "a": out_a,
        "I_S": out_I_S,
    }


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
_SEPARATOR = "─" * 78


def _time_fn(fn, n_repeats: int) -> list[float]:
    """Return a list of elapsed-wall-clock seconds for *n_repeats* calls."""
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return times


def _fmt_time(seconds: float) -> str:
    """Human-friendly time formatting."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:8.2f} ms"
    else:
        return f"{seconds:8.4f}  s"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

# Timepoint counts to test  (256 → 32 768 covers "several thousand")
BENCHMARK_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
N_REPEATS = 5  # runs per size (median is reported)

# Common model parameters
PARAMS = dict(d=0.0, taup=1.0, taug=1.0, B=0.05, Cm=0.0, toff=0.0)
TM = np.array([7.0, 11.0, 17.5])


def benchmark():
    """Run the full benchmark suite."""
    print()
    print(
        "╔══════════════════════════════════════════════════════════════════════════════╗"
    )
    print(
        "║         PFun CMA Engine — Python vs. C Extension Benchmark                 ║"
    )
    print(
        "╚══════════════════════════════════════════════════════════════════════════════╝"
    )
    print()
    print(f"  Library:   {_lib._name}")
    print(f"  Repeats:   {N_REPEATS} per configuration (median reported)")
    print(f"  Sizes:     {BENCHMARK_SIZES}")
    print(f"  Params:    {PARAMS}")
    print()
    print(_SEPARATOR)

    results: list[dict] = []

    for N in BENCHMARK_SIZES:
        t = np.linspace(0, 24, N)

        # ── Python engine ──────────────────────────────────────────────
        model = CMASleepWakeModel(N=N, tM=TM)
        model.update(**PARAMS)

        def _run_python():
            model.update(**PARAMS)
            _ = model.g_instant  # forces full pipeline evaluation

        py_times = _time_fn(_run_python, N_REPEATS)
        py_median = statistics.median(py_times)

        # ── C engine ──────────────────────────────────────────────────
        def _run_c():
            res = _run_c_engine(
                t,
                d=PARAMS["d"],
                taup=PARAMS["taup"],
                taug_val=PARAMS["taug"],
                B=PARAMS["B"],
                Cm=PARAMS["Cm"],
                toff=PARAMS["toff"],
                tM=TM,
            )
            _ = res["G"]  # ensure output is materialised

        c_times = _time_fn(_run_c, N_REPEATS)
        c_median = statistics.median(c_times)

        speedup = py_median / c_median if c_median > 0 else float("inf")

        # ── Correctness check ──────────────────────────────────────────
        py_G = model.g_instant
        c_res = _run_c_engine(
            t,
            d=PARAMS["d"],
            taup=PARAMS["taup"],
            taug_val=PARAMS["taug"],
            B=PARAMS["B"],
            Cm=PARAMS["Cm"],
            toff=PARAMS["toff"],
            tM=TM,
        )
        # The C engine's out_G already includes the bias term.
        c_G = c_res["G"]
        max_diff = float(np.abs(py_G - c_G).max())

        results.append(
            {
                "N": N,
                "py_median": py_median,
                "c_median": c_median,
                "speedup": speedup,
                "max_diff": max_diff,
            }
        )

        status = "✓" if max_diff < 1e-6 else "✗"
        print(
            f"  N={N:>6,}  │  Python {_fmt_time(py_median)}  │  "
            f"C {_fmt_time(c_median)}  │  "
            f"{speedup:6.1f}×  │  Δmax={max_diff:.2e} {status}"
        )

    print(_SEPARATOR)
    print()

    # ── Summary table ─────────────────────────────────────────────────
    print("  Summary Table")
    print("  ┌──────────┬──────────────┬──────────────┬──────────┬─────────────┐")
    print("  │  N       │  Python      │  C Extension │  Speedup │  Max |Δ|    │")
    print("  ├──────────┼──────────────┼──────────────┼──────────┼─────────────┤")
    for r in results:
        print(
            f"  │ {r['N']:>7,} │ {_fmt_time(r['py_median'])} │ "
            f"{_fmt_time(r['c_median'])} │ {r['speedup']:>6.1f}× │ "
            f"{r['max_diff']:>9.2e}   │"
        )
    print("  └──────────┴──────────────┴──────────────┴──────────┴─────────────┘")

    # ── Aggregate stats ──────────────────────────────────────────────
    avg_speedup = statistics.mean(r["speedup"] for r in results)
    max_speedup = max(r["speedup"] for r in results)
    total_py = sum(r["py_median"] for r in results)
    total_c = sum(r["c_median"] for r in results)
    all_correct = all(r["max_diff"] < 1e-6 for r in results)

    print()
    print(f"  Average speedup:  {avg_speedup:.1f}×")
    print(
        f"  Peak speedup:     {max_speedup:.1f}×  (N={max(results, key=lambda r: r['speedup'])['N']:,})"
    )
    print(f"  Total Python:     {_fmt_time(total_py)}")
    print(f"  Total C:          {_fmt_time(total_c)}")
    print(
        f"  Correctness:      {'ALL PASSED ✓' if all_correct else 'MISMATCH DETECTED ✗'}"
    )
    print()

    return results


if __name__ == "__main__":
    benchmark()
