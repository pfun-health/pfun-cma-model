---
icon: lucide/rocket
---

# PFun CMA Model

<div align="center">

![PFun Logo](assets/img/pfunCelticFlower.png){ width="180" }

**A physiological chronometabolic model for glucose, hormone, and circadian rhythm analysis.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/pfun-health/pfun-cma-model)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688.svg)](https://fastapi.tiangolo.com)

[Live Demo](https://cloud.tail38611b.ts.net/demo/llm){ .md-button .md-button--primary }
[API Docs](api.md){ .md-button }
[Source Code](https://github.com/pfun-health/pfun-cma-model){ .md-button }

</div>

---

## What is PFun CMA?

The **PFun CMA Model** is a generative physiological model that functionally replicates neuroendocrine dynamics — specifically the interplay between **cortisol**, **melatonin**, and **adiponectin** — and their influence on glucose metabolism across the circadian cycle.

### In simple terms

- :material-compress: **Phase-based dimensionality reduction** — Compress weeks or months of CGM time-series data into a compact phase vector (`≥ 1024b in memory`).
- :material-translate: **Interpretable & Quantifiable** — Translate between qualitative states ("mood", "stress") and biophysical neuroendocrine dynamics ("cortisol levels", "glucose variability").
- :material-clock-fast: **High-speed circadian mapping** — Understand how circadian rhythm maps to glucose values in real time.

---

## Generated CMA Decomposition

The model decomposes glucose time series data into underlying hormonal influences:

![Generated Cortisol-Melatonin-Adiponectin decomposition from Glucose time series](assets/img/generated.png)

> The CMA model leverages physiological modeling principles to decompose glucose time series data into underlying hormonal influences — specifically cortisol, melatonin, and adiponectin.

---

## Model Fitting

Fit the CMA model to real CGM data and extract clinically meaningful parameters:

![24-hour fit result — blue is model, red is data](assets/img/24hr_fit_result_blueModel_redData.png)

After fitting, visualize parameters with automatically generated qualitative descriptions:

![Fit result with qualitative descriptions](assets/img/24hr_fit_result_pretty_markdown_table_with_qualitative_descriptions.png)

---

## Live Demos

### LLM-Powered Scenario Generation

Generate realistic physiological scenarios using natural language prompts. The LLM translates qualitative descriptions into physiologically valid CMA parameters:

![LLM Generate Scenario Demo](assets/img/Screenshot 2026-02-13 at 14-25-06 LLM Generate Scenario Demo ~ PFun Digital Health.png)

![LLM Scenario with recommendations](assets/img/Screenshot 2026-02-13 at 14-28-01 LLM Generate Scenario Demo ~ PFun Digital Health.png)

---

### Real-time WebSocket Streaming

Interactive parameter control with live chart updates via WebSocket:

![WebSocket streaming demo](assets/img/Screenshot 2025-08-10 at 02-36-51 Run-at-Time WebSocket Example.png)

![WebSocket demo with sliders](assets/img/Screenshot 2025-08-11 at 18-53-40 Run-at-Time WebSocket Example.png)

---

### Parameter Grid Visualization

Explore the CMA parameter space with precomputed grids:

![CMA parameter grid visualization](assets/img/Screenshot_20260416_201527 cma_grid cma_pgrid.png)

---

## Quick Start

```bash
# Install
pip install pfun-cma-model
# — or with uv —
uv add pfun-cma-model

# Run a quick model fit
uv run pfun-cma-model fit-model --plot

# Generate a scenario via LLM
uv run pfun-cma-model generate-scenario \
  --query "a healthy individual who exercises before sunrise"

# Launch the full dev server
uv run fastapi dev pfun_cma_model/app.py --port 8001
```

**→ [Full installation guide](getting-started/installation.md)**

---

## Project Architecture

```
pfun-cma-model/
├── pfun_cma_model/          # Core Python package
│   ├── engine/              # CMA model, fit, grid, plotting
│   │   ├── cma.py           # CMASleepWakeModel class
│   │   ├── fit.py           # Curve fitting pipeline
│   │   ├── grid.py          # Parameter grid search
│   │   └── pfun_cma_engine.c  # Optional C extension
│   ├── routes/              # FastAPI route handlers
│   │   ├── demo.py          # Interactive demo endpoints
│   │   ├── llm.py           # LLM scenario generation API
│   │   ├── ws.py            # WebSocket streaming
│   │   └── params.py        # Parameter CRUD endpoints
│   ├── llm.py               # LLM prompting logic
│   ├── cli.py               # Click CLI commands
│   └── security.py          # Security middleware
├── packages/
│   ├── pfun_common/         # Shared utilities & settings
│   └── pfun_qt_gui/         # Qt6 desktop application
├── examples/
│   ├── notebooks/           # Jupyter notebooks
│   ├── screenshots/         # Application screenshots
│   └── videos/              # Demo recordings
├── docs/                    # This documentation (Zensical)
├── tests/                   # Test suite
└── results/                 # Output artifacts & databases
```

---

## Key Links

| Resource | Link |
|----------|------|
| :material-web: Homepage | [pfun.one](https://pfun.one/) |
| :material-play-circle: Live Demo | [PFun Health Tips](https://cloud.tail38611b.ts.net/demo/llm) |
| :material-github: Source Code | [pfun-health/pfun-cma-model](https://github.com/pfun-health/pfun-cma-model) |
| :material-file-document: Research Paper | [Chronometabolic Analysis (PDF)](https://github.com/pfun-health/pfun-cma-model/blob/main/docs/rendered_pdf/PFun%20Glucose%20-%20Chronometabolic%20Analysis.pdf) |
| :material-api: API Swagger | `http://localhost:8001/docs` |
| :material-api: API ReDoc | `http://localhost:8001/redoc` |
