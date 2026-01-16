# Copilot Instructions for pfun-cma-model

## Project Overview
- __Domain:__ Circadian and metabolic modeling, with a focus on glucose and cortisol dynamics.
- __Core:__ The `pfun_cma_model` package implements the main model logic, CLI, and FastAPI app.
- __Data Flow:__ Model parameters are defined, fitted, and interpreted; results are visualized and output to files in `output/` and `results/`.
- __Key Directories:__
  - `pfun_cma_model/`: Main model, API, CLI, and engine logic
  - `pfun_common/`, `pfun_data/`: Shared utilities and data helpers
  - `examples/`: Scripts and notebooks for demos, parameter interpretation, and UI
  - `tests/`: Pytest-based test suite

## Developer Workflows
- __Environment:__ Use `uv` for dependency management and running commands. Create a venv with `uv venv`.
- __Install dependencies:__ `uv sync` (syncs with lock files)
- __Run dev server:__ `uv run fastapi dev pfun_cma_model/app.py --port 8001`
- __Run CLI:__ `uv run pfun-cma-model` (shows usage)
- __Fit model:__ `uv run pfun-cma-model run-fit-model --plot`
- __Run tests:__ `pytest` or `uvx pytest`
- __Add dev dependency:__ `uv add --dev <package>`

## Project Conventions & Patterns
- __Parameter schemas:__ Defined in `pfun_cma_model/engine/` and described in README tables.
- __Notebooks:__ `notebooks/` and `examples/` provide usage, visualization, and parameter interpretation.
- __Output:__ Model results and plots are written to `output/` and `results/`.
- __Testing:__ All tests are in `tests/`, use `pytest` for running.
- __CLI/Server:__ Both CLI and FastAPI server entrypoints are in `pfun_cma_model/`.
- __Data:__ Example and training data in `examples/data/`.

## Integration & Extensibility
- __OpenAPI:__ `openapi.json` and scripts in `scripts/` for client generation.
- __Dash UI:__ Example Dash UI in `examples/dash_ui/`.
- __Docker:__ `Dockerfile` and `docker-compose.yaml` for containerization.

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
_Last updated: 2026-01-15_