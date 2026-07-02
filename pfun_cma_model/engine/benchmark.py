#!/usr/bin/env python
"""Benchmark: Python CMA engine vs. C extension.

Produces several thousand timepoints and compares wall-clock performance
of the pure-Python CMASleepWakeModel against the compiled C extension
(libpfun_cma_engine.so).

Usage:
    python -m pfun_cma_model.engine.benchmark          # from project root
    python pfun_cma_model/engine/benchmark.py           # direct invocation
    python pfun_cma_model/engine/benchmark.py --help    # show help
"""

import argparse
import ctypes
import json
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


def _load_c_library() -> Optional[ctypes.CDLL]:
    """Load the C shared library, returning None if not found."""
    for candidate in _LIB_CANDIDATES:
        if candidate.exists():
            return ctypes.CDLL(str(candidate))
    return None


_lib = _load_c_library()

# Regardless of whether C lib was found, we defer the hard error to when
# C is actually requested. This allows Python-only benchmarks to work
# without the C library installed.

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


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Python CMA engine vs. C extension",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pfun_cma_model.engine.benchmark
  python pfun_cma_model/engine/benchmark.py --sizes 1000,2000,5000 --repeats 10
  python pfun_cma_model/engine/benchmark.py --implementations python --output results.json
        """
    )
    
    parser.add_argument(
        "--sizes",
        type=str,
        default="256,512,1024,2048,4096,8192,16384,32768",
        help="Comma-separated list of timepoint counts to test (default: 256,512,1024,2048,4096,8192,16384,32768)"
    )
    
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of runs per configuration (median reported) (default: 5)"
    )
    
    parser.add_argument(
        "--implementations",
        type=str,
        default="python,c",
        help="Comma-separated list of implementations to test (python, c) (default: python,c)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save results in JSON format"
    )
    
    parser.add_argument(
        "--params",
        type=str,
        help="JSON string of model parameters to use (default: d=0.0, taup=1.0, taug=1.0, B=0.05, Cm=0.0, toff=0.0)"
    )
    
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs before timing begins (default: 3)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Maximum allowed absolute difference for correctness check (default: 1e-6)"
    )

    return parser.parse_args()


def _parse_sizes(sizes_str: str) -> list[int]:
    """Parse comma-separated sizes string into list of integers."""
    try:
        return [int(x.strip()) for x in sizes_str.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"Invalid sizes format: {sizes_str}. Expected comma-separated integers.") from e


def _parse_implementations(impl_str: str) -> list[str]:
    """Parse comma-separated implementations string into list of strings."""
    valid_impls = {"python", "c"}
    impls = [x.strip().lower() for x in impl_str.split(",") if x.strip()]
    
    invalid_impls = set(impls) - valid_impls
    if invalid_impls:
        raise ValueError(f"Invalid implementations: {invalid_impls}. Valid options: {valid_impls}")
    
    return impls


def _parse_params(params_str: str) -> dict:
    """Parse JSON string of parameters."""
    if not params_str:
        return {}
    try:
        return json.loads(params_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in params: {params_str}") from e


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
_SEPARATOR = "─" * 78


def _time_fn(fn, n_repeats: int, n_warmup: int = 0) -> list[float]:
    """Return a list of elapsed-wall-clock seconds for *n_repeats* calls.
    
    Args:
        fn: Function to time.
        n_repeats: Number of timed runs.
        n_warmup: Number of untimed warmup runs to prime caches/JIT.
    """
    # Warmup runs (excluded from measurements)
    for _ in range(n_warmup):
        fn()
    
    # Timed runs
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


def benchmark(sizes=None, n_repeats=None, implementations=None, custom_params=None, n_warmup=3, threshold=1e-6):
    """Run the full benchmark suite.
    
    Args:
        sizes: List of timepoint counts to test. Defaults to BENCHMARK_SIZES.
        n_repeats: Number of runs per configuration. Defaults to N_REPEATS.
        implementations: List of implementations to test ('python', 'c'). Defaults to both.
        custom_params: Dictionary of model parameters to use. Defaults to PARAMS.
        n_warmup: Number of warmup runs before timing begins. Defaults to 3.
        threshold: Maximum allowed absolute difference for correctness check. Defaults to 1e-6.
    """
    # Use defaults if not provided
    if sizes is None:
        sizes = BENCHMARK_SIZES
    if n_repeats is None:
        n_repeats = N_REPEATS
    if implementations is None:
        implementations = ['python', 'c']
    if custom_params is None:
        params = PARAMS
    else:
        params = dict(PARAMS)  # Copy defaults
        params.update(custom_params)  # Override with custom params
    
    # Validate inputs
    if not sizes:
        raise ValueError("sizes list must not be empty")
    for s in sizes:
        if s < 1:
            raise ValueError(f"Invalid size: {s}. Must be >= 1")
    if n_repeats < 1:
        raise ValueError(f"Invalid repeats: {n_repeats}. Must be >= 1")
    if n_warmup < 0:
        raise ValueError(f"Invalid warmup: {n_warmup}. Must be >= 0")
    if threshold <= 0:
        raise ValueError(f"Invalid threshold: {threshold}. Must be > 0")
    if 'c' in implementations and _lib is None:
        print(
            f"ERROR: C implementation requested but could not find libpfun_cma_engine.so.\n"
            f"Searched: {[str(c) for c in _LIB_CANDIDATES]}\n"
            f"Build it first:  cd pfun-cma-engine-c && make",
            file=sys.stderr,
        )
        sys.exit(1)
    
    required_params = {"d", "taup", "B", "Cm", "toff", "taug"}
    missing_params = required_params - set(params.keys())
    if missing_params:
        raise ValueError(f"Missing required model parameters: {missing_params}")
    
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
    print(f"  Library:   {_lib._name if _lib else 'N/A'}")
    print(f"  Repeats:   {n_repeats} per configuration (median reported)")
    print(f"  Warmup:    {n_warmup} runs excluded from timing")
    print(f"  Sizes:     {sizes}")
    print(f"  Params:    {params}")
    print(f"  Testing:   {', '.join(implementations)}")
    print()
    print(_SEPARATOR)

    results: list[dict] = []

    for N in sizes:
        t = np.linspace(0, 24, N)

        # Initialize timing results
        py_times = []
        c_times = []

        # ── Python engine ──────────────────────────────────────────────
        if 'python' in implementations:
            model = CMASleepWakeModel(N=N, tM=TM)
            model.update(**params)

            def _run_python():
                model.update(**params)
                _ = model.g_instant  # forces full pipeline evaluation

            py_times = _time_fn(_run_python, n_repeats, n_warmup)
            py_median = statistics.median(py_times)
            py_mean = statistics.mean(py_times)
            py_min = min(py_times)
            py_max = max(py_times)
            if len(py_times) > 1:
                py_stdev = statistics.stdev(py_times)
            else:
                py_stdev = 0.0
        else:
            py_median = py_mean = py_min = py_max = py_stdev = 0.0

        # ── C engine ──────────────────────────────────────────────────
        if 'c' in implementations and _lib is not None:
            def _run_c():
                res = _run_c_engine(
                    t,
                    d=params["d"],
                    taup=params["taup"],
                    taug_val=params["taug"],
                    B=params["B"],
                    Cm=params["Cm"],
                    toff=params["toff"],
                    tM=TM,
                )
                _ = res["G"]  # ensure output is materialised

            c_times = _time_fn(_run_c, n_repeats, n_warmup)
            c_median = statistics.median(c_times)
            c_mean = statistics.mean(c_times)
            c_min = min(c_times)
            c_max = max(c_times)
            if len(c_times) > 1:
                c_stdev = statistics.stdev(c_times)
            else:
                c_stdev = 0.0
        else:
            c_median = c_mean = c_min = c_max = c_stdev = 0.0

        # ── Speedup calculation ────────────────────────────────────────
        speedup = py_median / c_median if c_median > 0 and py_median > 0 else float("inf")

        # ── Correctness check ──────────────────────────────────────────
        max_diff = 0.0
        if 'python' in implementations and 'c' in implementations and _lib is not None:
            py_G = model.g_instant
            c_res = _run_c_engine(
                t,
                d=params["d"],
                taup=params["taup"],
                taug_val=params["taug"],
                B=params["B"],
                Cm=params["Cm"],
                toff=params["toff"],
                tM=TM,
            )
            # The C engine's out_G already includes the bias term.
            c_G = c_res["G"]
            max_diff = float(np.abs(py_G - c_G).max())

        # ── Store results ──────────────────────────────────────────────
        result = {
            "N": N,
            "max_diff": max_diff,
        }
        
        if 'python' in implementations:
            result.update({
                "py_median": py_median,
                "py_mean": py_mean,
                "py_min": py_min,
                "py_max": py_max,
                "py_stdev": py_stdev,
            })
            
        if 'c' in implementations and _lib is not None:
            result.update({
                "c_median": c_median,
                "c_mean": c_mean,
                "c_min": c_min,
                "c_max": c_max,
                "c_stdev": c_stdev,
                "speedup": speedup,
            })

        results.append(result)

        # ── Print results ──────────────────────────────────────────────
        status = "✓" if max_diff < threshold else "✗"
        
        if 'python' in implementations and 'c' in implementations and _lib is not None:
            print(
                f"  N={N:>6,}  │  Python {_fmt_time(py_median)}  │  "
                f"C {_fmt_time(c_median)}  │  "
                f"{speedup:6.1f}×  │  Δmax={max_diff:.2e} {status}"
            )
        elif 'python' in implementations:
            print(
                f"  N={N:>6,}  │  Python {_fmt_time(py_median)}  │  "
                f"{'N/A':>12}  │  "
                f"{'N/A':>7}  │  Δmax={'N/A':>9}   "
            )
        elif 'c' in implementations and _lib is not None:
            print(
                f"  N={N:>6,}  │  {'N/A':>12}  │  "
                f"C {_fmt_time(c_median)}  │  "
                f"{'N/A':>7}  │  Δmax={'N/A':>9}   "
            )

    print(_SEPARATOR)
    print()

    # ── Summary table ─────────────────────────────────────────────────
    print("  Summary Table")
    
    if 'python' in implementations and 'c' in implementations and _lib is not None:
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
    elif 'python' in implementations:
        print("  ┌──────────┬──────────────┐")
        print("  │  N       │  Python      │")
        print("  ├──────────┼──────────────┤")
        for r in results:
            print(
                f"  │ {r['N']:>7,} │ {_fmt_time(r['py_median'])} │"
            )
        print("  └──────────┴──────────────┘")
    elif 'c' in implementations and _lib is not None:
        print("  ┌──────────┬──────────────┐")
        print("  │  N       │  C Extension │")
        print("  ├──────────┼──────────────┤")
        for r in results:
            print(
                f"  │ {r['N']:>7,} │ {_fmt_time(r['c_median'])} │"
            )
        print("  └──────────┴──────────────┘")

    # ── Aggregate stats ──────────────────────────────────────────────
    if 'python' in implementations and 'c' in implementations and _lib is not None:
        avg_speedup = statistics.mean(r["speedup"] for r in results if "speedup" in r)
        max_speedup = max(r["speedup"] for r in results if "speedup" in r)
        total_py = sum(r["py_median"] for r in results if "py_median" in r)
        total_c = sum(r["c_median"] for r in results if "c_median" in r)
        all_correct = all(r["max_diff"] < threshold for r in results)
        
        print()
        print(f"  Average speedup:  {avg_speedup:.1f}×")
        print(
            f"  Peak speedup:     {max_speedup:.1f}×  (N={max(results, key=lambda r: r.get('speedup', 0))['N']:,})"
        )
        print(f"  Total Python:     {_fmt_time(total_py)}")
        print(f"  Total C:          {_fmt_time(total_c)}")
        print(
            f"  Correctness:      {'ALL PASSED ✓' if all_correct else 'MISMATCH DETECTED ✗'}"
        )
    elif 'python' in implementations:
        total_py = sum(r["py_median"] for r in results if "py_median" in r)
        print()
        print(f"  Total Python time: {_fmt_time(total_py)}")
    elif 'c' in implementations and _lib is not None:
        total_c = sum(r["c_median"] for r in results if "c_median" in r)
        print()
        print(f"  Total C time:      {_fmt_time(total_c)}")

    print()

    return results


if __name__ == "__main__":
    args = _parse_args()
    
    # Parse arguments
    try:
        benchmark_sizes = _parse_sizes(args.sizes)
        implementations = _parse_implementations(args.implementations)
        custom_params = _parse_params(args.params)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run benchmark with parsed arguments
    results = benchmark(
        sizes=benchmark_sizes,
        n_repeats=args.repeats,
        implementations=implementations,
        custom_params=custom_params,
        n_warmup=args.warmup,
        threshold=args.threshold,
    )
    
    # Output results to file if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        except Exception as e:
            print(f"Error saving results to {args.output}: {e}", file=sys.stderr)
