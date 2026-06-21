# DEMO — PFun CMA Model: Terminal Video Generator

A terminal-based demo video generator that showcases the PFun CMA Model's performance and visualization capabilities.

## Quick Start

[Source: `demo-video.py`](scripts/demo-video.py)

```bash
uv run python scripts/demo-video.py
```

## Recording

```bash
# Terminal recording (no GUI needed)
asciinema rec -c "uv run python scripts/demo-video.py" demo.cast

# Screen capture (GUI session, requires display)
ffmpeg -f x11grab -i :0.0 -framerate 10 cma-demo.mp4
```

---

## Demo 1: Vectorized Operations on Parameter Column

Demonstrates the performance advantage of storing CMA model output in dense **vectorized columns** in DuckDB versus the existing sparse **JSON metadata** approach.

| Metric | Result |
|--------|--------|
| Dense column scan (1024 pts) | ~1.5 ms |
| Statistics computation (MIN, MAX, AVG, STDDEV) | ~8 ms |
| JSON parsing + extraction (sparse) | ~35 ms |
| **Speedup** | **4.3x faster** |

## Demo 2: 3D Waveform Visualization in Terminal

Animates four physiological signals across a 24-hour circadian cycle using pure ASCII characters in the terminal:

- **Cortisol (c)** `●` — Dawn-peaking stress hormone
- **Melatonin (m)** `○` — Darkness-activated sleep signal
- **Adiponectin (a)** `▲` — Insulin sensitivity modulator
- **Glucose (G)** `■` — Post-prandial metabolic output

**Frame count**: 30 frames (0.1s delay each)
**Data points per frame**: 1024
**Animation**: Phase-rolling time window across the circadian cycle

### Sample Render

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ M Melatonin (m) : ○                                                          │
│ A Adiponectin (a): ▲○          ●●                                            │
│ G Glucose (G)   : ■ ○○○      ●●● ●●●●●                                       │
│           ○○○         ○○     ●       ●●●●                                    │
│          ○○            ○○   ●●           ●●●●                                │
│       ○○                ○   ●               ●●●●●                            │
│     ○○○                 ○○ ●●                 ▲▲▲▲▲▲▲▲▲                      │
│    ○○                    ○ ●              ▲▲▲▲▲     ●●▲▲▲▲                   │
│  ○○○                     ○○●           ▲▲▲▲             ●●▲▲●                │
│ ○○                        ○●         ▲▲▲                    ▲▲●●●            │
│○○                         ●       ▲▲▲▲                        ▲▲●●●          │
│○                          ○○   ▲▲▲▲                            ▲▲▲●●        ○│
│                           ●○▲▲▲▲                                 ▲▲▲●●     ○○│
│                          ▲▲▲▲                                      ▲▲●    ○○ │
│                    ▲▲▲▲▲▲●  ○                                        ▲▲▲▲○○  │
│▲▲▲▲▲▲▲▲▲ ▲▲▲▲▲▲▲▲▲▲▲     ●  ○○                                         ●▲▲▲▲▲│
│                         ●●   ○                                         ○●●●  │
│●                        ●    ○○                                       ○○  ●●●│
│●●●●●●    ●●            ●●     ○○                                    ○○○      │
│     ●●●●  ●●●●●●●●●●●●●●       ○○○                                ○○○        │
│                                  ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○           │
│■■■■■■■■ ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■│
│──────────────────────────────────────────────────────────────────────────────│
└──────────────────────────────────────────────────────────────────────────────┘
```

## Demo 3: DuckDB vs Pure NumPy Performance

Compares vectorized aggregation performance between DuckDB and NumPy on the 1024-point dense dataset.

| Operation | DuckDB | NumPy | Ratio |
|-----------|--------|-------|-------|
| AVG+STDDEV+MIN+MAX (1024 pts) | 0.74 ms | 0.05 ms | 14x |
| Same operations on 10,000 pts | — | 0.14 ms | — |

## Demo 4: HTML Visualization Output

Generates a standalone HTML page (`dist/demo-visualization.html`) containing:

- SVG line charts of all four signals
- Summary statistics table
- Model parameter table
- Dark-themed UI with monospace styling

---

## Output Files

| Path | Description |
|------|-------------|
| `results/cma_dense.db` | DuckDB with 1024-row dense table (12 signal columns), 81-param grid (binary-packed arrays), and sample params |
| `dist/demo-visualization.html` | Standalone HTML visualization with SVG signal plots |
| `scripts/demo-video.py` | The demo generator script |

## Technical Details

- **CMA Model**: `CMASleepWakeModel(N=1024)` generates 1024 time points across a 24-hour cycle
- **DuckDB Schema**:
  - `cma_dense` — id, t, c, m, a, I_S, I_E, L, g_0, g_1, g_2, G (12 columns, 1024 rows)
  - `cma_pgrid_dense` — id, B, Cm, taug, taup, t, c, m, a, G (binary-packed BLOBs, 81 rows)
  - `cma_params` — id, d, taup, taug, B, Cm, toff (3 sample parameter sets)
- **Terminal rendering**: 80×25 character buffer with box-drawing Unicode characters
- **Animation**: Phase-shifted time window, 30 frames at 0.1s intervals
