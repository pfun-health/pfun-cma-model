#!/usr/bin/env python3
"""Generate golden curves from the public Python CMA engine (pfun_cma_model.engine).

The WebAssembly engine used by this demo implements the same circadian and glucose
equations as ``pfun_cma_model.engine.cma.CMASleepWakeModel`` (plus an activity term
that the Python engine does not have).  ``test/engine.test.ts`` runs the wasm engine
without activity on the parameter sets below and requires it to reproduce these
curves, so the demo cannot silently drift from the model this repository documents.

Run from anywhere:

    python3 cloudflare/time2move/scripts/gen-python-fixtures.py

It needs numpy, pandas and pydantic.  The package's top-level ``__init__`` pulls in
optional heavy dependencies (numba, scipy, ...); when those are not installed the
engine modules are imported directly instead, which is all this script uses.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT = HERE.parents[1] / "test" / "fixtures" / "python-engine.json"

N_POINTS = 49  # half-hour grid over one day

# name -> CMASleepWakeModel keyword arguments (taug is a scalar applied to every meal)
CASES: dict[str, dict] = {
    "python-defaults": dict(d=0.0, taup=1.0, taug=1.0, B=0.05, Cm=0.0, toff=0.0, tM=[7.0, 11.0, 17.5]),
    "engine-calibration": dict(d=1.0, taup=1.2, taug=1.0, B=0.05, Cm=0.0, toff=0.0, tM=[7.0, 11.0, 17.5]),
    "late-chronotype-two-meals": dict(d=2.5, taup=1.6, taug=1.3, B=0.08, Cm=0.0, toff=0.0, tM=[11.0, 19.5]),
    "raised-basal-broad-meals": dict(d=1.0, taup=1.02, taug=1.5, B=0.1, Cm=0.0, toff=0.0, tM=[6.5, 12.0, 18.0]),
    "cortisol-coupled-bias": dict(d=-0.5, taup=0.8, taug=0.9, B=0.12, Cm=0.6, toff=1.5, tM=[8.0, 13.0, 20.0]),
}

SIGNALS = ("L", "m", "c", "a", "I_S", "I_E")


def import_engine():
    sys.path.insert(0, str(REPO))
    try:
        from pfun_cma_model.engine.cma import CMASleepWakeModel  # noqa: PLC0415

        return CMASleepWakeModel
    except ModuleNotFoundError:
        # Skip pfun_cma_model/__init__.py (optional heavy deps) and tabulate (only
        # used to pretty-print parameter tables).
        for name in [m for m in sys.modules if m == "pfun_cma_model" or m.startswith("pfun_cma_model.")]:
            del sys.modules[name]
        for pkg in ("pfun_cma_model", "pfun_cma_model.engine", "pfun_cma_model.misc"):
            module = types.ModuleType(pkg)
            module.__path__ = [str(REPO / pkg.replace(".", "/"))]
            sys.modules[pkg] = module
        if "tabulate" not in sys.modules:
            try:
                import tabulate  # noqa: F401, PLC0415
            except ModuleNotFoundError:
                stub = types.ModuleType("tabulate")
                stub.tabulate = lambda *a, **k: ""
                sys.modules["tabulate"] = stub
        from pfun_cma_model.engine.cma import CMASleepWakeModel  # noqa: PLC0415

        return CMASleepWakeModel


def main() -> None:
    CMASleepWakeModel = import_engine()
    cases = {}
    for name, kwargs in CASES.items():
        model = CMASleepWakeModel(N=N_POINTS, **kwargs)
        curves = {key: [float(v) for v in getattr(model, key)] for key in SIGNALS}
        curves["G"] = [float(v) for v in model.g_instant]
        cases[name] = {"params": kwargs, "t": [float(v) for v in model.t], **curves}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generator": "cloudflare/time2move/scripts/gen-python-fixtures.py",
                "engine": "pfun_cma_model.engine.cma.CMASleepWakeModel",
                "cases": cases,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(cases)} cases x {N_POINTS} points)")


if __name__ == "__main__":
    main()
