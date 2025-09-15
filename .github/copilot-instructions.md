# Copilot Instructions for pfun-cma-model

## Project Overview
- **Domain:** Circadian and metabolic modeling, with a focus on glucose and cortisol dynamics.
- **Core:** The `pfun_cma_model` package implements the main model logic, CLI, and FastAPI app.
- **Data Flow:** Model parameters are defined, fitted, and interpreted; results are visualized and output to files in `output/` and `results/`.
- **Key Directories:**
  - `pfun_cma_model/`: Main model, API, CLI, and engine logic
  - `pfun_common/`, `pfun_data/`: Shared utilities and data helpers
  - `examples/`: Scripts and notebooks for demos, parameter interpretation, and UI
  - `tests/`: Pytest-based test suite

## Developer Workflows
- **Environment:** Use `uv` for dependency management and running commands. Create a venv with `uv venv`.
- **Install dependencies:** `uv sync` (syncs with lock files)
- **Run dev server:** `uv run fastapi dev pfun_cma_model/app.py --port 8001`
- **Run CLI:** `uv run pfun-cma-model` (shows usage)
- **Fit model:** `uv run pfun-cma-model run-fit-model --plot`
- **Run tests:** `uvx tox`
- **Add dev dependency:** `uv add --dev <package>`

## Project Conventions & Patterns
- **Parameter schemas:** Defined in `pfun_cma_model/engine/` and described in README tables.
- **Notebooks:** `notebooks/` and `examples/` provide usage, visualization, and parameter interpretation.
- **Output:** Model results and plots are written to `output/` and `results/`.
- **Testing:** All tests are in `tests/`, use `pytest` via `tox` or `uvx tox`.
- **CLI/Server:** Both CLI and FastAPI server entrypoints are in `pfun_cma_model/`.
- **Data:** Example and training data in `examples/data/`.

## Integration & Extensibility
- **OpenAPI:** `openapi.json` and scripts in `scripts/` for client generation.
- **Dash UI:** Example Dash UI in `examples/dash_ui/`.
- **Docker:** `Dockerfile` and `docker-compose.yaml` for containerization.

## Examples
- See `examples/` for scripts like `generate-n-samples.py`, `interpret-cma-params.py`.
- Notebooks in `notebooks/` for demos and visualization.

## Tips for AI Agents
- Prefer `uv` for all Python environment and run commands.
- Reference README for parameter details and workflow examples.
- Use existing scripts/notebooks as templates for new features or analyses.
- Keep outputs in `output/` or `results/` for consistency.
- Follow the structure of `pfun_cma_model/engine/` for new model logic.

---
_Last updated: 2025-09-13_
