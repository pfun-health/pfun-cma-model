# AGENTS.md - Developer Guide for pfun-cma-model

This file provides guidance for agentic coding agents working in this repository.

## Project Overview

- **Python version**: 3.12.11
- **Package manager**: `uv`
- **Test framework**: pytest
- **Linter**: ruff
- **Type checker**: mypy
- **Formatter**: ruff

## Key References

### Core Package
- [`README.md`](./README.md) - Project overview
- [`pyproject.toml`](./pyproject.toml) - Package configuration and dependencies
- [`pfun_cma_model/app.py`](./pfun_cma_model/app.py) - Central FastAPI application
- [`packages/pfun_common/pyproject.toml`](./packages/pfun_common/pyproject.toml) - Common utilities

### Qt GUI Package
- [`packages/pfun_qt_gui/pyproject.toml`](./packages/pfun_qt_gui/pyproject.toml) - Qt GUI package
- [`packages/pfun_qt_gui/.env`](./packages/pfun_qt_gui/.env) - Qt GUI environment variables

### Scripts
- [`scripts/launch-qt-gui.sh`](./scripts/launch-qt-gui.sh) - Launch Qt GUI script
- [`scripts/uv-full-sync.sh`](./scripts/uv-full-sync.sh) - Full uv sync script
- [`scripts/_funcs.def.sh`](./scripts/_funcs.def.sh) - Common functions

---

## Build, Lint, and Test Commands

### Environment Setup

```bash
# Create virtual environment
uv venv

# Sync dependencies
uv sync

# Activate (if using venv directly)
source .venv/bin/activate
```

### Running the Application

```bash
# Run FastAPI dev server
uv run fastapi dev pfun_cma_model/app.py --port 8001

# Run via CLI
uv run pfun-cma-model COMMAND [ARGS]
```

### Testing

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_fit.py

# Run a single test function
uv run pytest tests/test_fit.py::test_estimate_mealtimes_invalid_input

# Run tests with verbose output
uv run pytest -v

# Run tests matching a pattern
uv run pytest -k "test_estimate"

# Run with coverage (if needed)
uv run pytest --cov=pfun_cma_model
```

### Linting and Formatting

```bash
# Run ruff linter
uv run ruff check .

# Run ruff with auto-fix
uv run ruff check --fix .

# Run ruff formatter
uv run ruff .

# Run mypy type checker
uv run mypy .
```

### Type Checking (mypy)

```bash
# Check types
uv run mypy pfun_cma_model

# Check specific file
uv run mypy pfun_cma_model/engine/fit.py
```

### Database Migrations

```bash
# Run Alembic migrations
uv run alembic upgrade head

# Create a new migration
uv run alembic revision --autogenerate -m "description"
```

---

## Code Style Guidelines

### General Principles

- Write clean, readable, and maintainable code
- Follow existing patterns in the codebase
- Use type hints for all function signatures
- Keep functions small and focused
- Write docstrings for public APIs

### Imports

- Use absolute imports (e.g., `from pfun_cma_model.engine.fit import ...`)
- Group imports in this order: stdlib, third-party, local
- Sort imports alphabetically within each group
- Use `__all__` to explicitly define public API

```python
# Good
import os
from typing import Optional

import pandas as pd
from fastapi import APIRouter

from pfun_cma_model.engine.fit import estimate_mealtimes
from pfun_cma_model.models import SomeModel
```

### Formatting

- Use **ruff** for code formatting (line length handled by ruff)
- Maximum line length: 88 characters (ruff default)
- Use 4 spaces for indentation (no tabs)
- Use trailing commas in multi-line structures
- One blank line between top-level definitions

### Type Hints

- Always use type hints for function parameters and return values
- Use `Optional[X]` instead of `X | None`
- Use `List`, `Dict`, `Tuple` from `typing` (or use lowercase `list`, `dict`, `tuple` for Python 3.9+)
- Be explicit about container types: `list[int]` not `list`

```python
# Good
def process_data(data: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    ...

# Good
def get_items() -> list[Item]:
    ...
```

### Naming Conventions

- **Variables/functions**: snake_case (e.g., `estimate_mealtimes`, `data_frame`)
- **Classes**: PascalCase (e.g., `CMAEngine`, `FitResult`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_ITERATIONS`)
- **Private members**: prefix with underscore (e.g., `_private_method`)
- **Files**: snake_case (e.g., `cma_model.py`)

### Docstrings

- Use Google-style docstrings for all public functions
- Include Args, Returns, and Raises sections when applicable

```python
def fit_model(data: pd.DataFrame, params: dict) -> FitResult:
    """Fit the CMA model to the provided data.

    Args:
        data: Time series data with glucose readings.
        params: Model parameters to optimize.

    Returns:
        Fitted model parameters and metrics.

    Raises:
        ValueError: If data is empty or params are invalid.
    """
```

### Error Handling

- Use custom exceptions for domain-specific errors
- Catch specific exceptions, not bare `Exception`
- Include meaningful error messages
- Use `pytest.raises` for testing exceptions

```python
# Good
def estimate_mealtimes(data: pd.DataFrame) -> list[float]:
    if data is None or data.empty:
        raise ValueError("Input data cannot be None or empty.")
    ...
```

### Async Code

- Use `async def` for I/O-bound operations
- Use `await` for async calls
- Use `run_in_executor` for blocking operations

### Database (SQLAlchemy)

- Use async SQLAlchemy with `AsyncSession`
- Always use dependency injection for database sessions
- Use Alembic for migrations

### Testing Guidelines

- Place tests in the `tests/` directory
- Name test files: `test_<module>.py`
- Test one thing per test function
- Use descriptive test names: `test_<function>_raises_on_invalid_input`
- Use fixtures for common test data

```python
import pytest
import pandas as pd

def test_estimate_mealtimes_invalid_input():
    with pytest.raises(ValueError, match="Input data cannot be None or empty."):
        estimate_mealtimes(None)
```

### Configuration

- Use Pydantic Settings for configuration
- Store secrets in `.env` (never commit)
- Use `.env.template` for required environment variables

---

## Project Structure

```
pfun-cma-model/
├── pfun_cma_model/       # Main package
│   ├── app.py            # FastAPI application
│   ├── cli.py            # CLI commands
│   ├── engine/           # C extension and engine code
│   └── ...
├── packages/pfun_common/  # Shared utilities
├── tests/                 # Test suite
├── alembic/              # Database migrations
└── pyproject.toml        # Project config
```

---

## Common Tasks

### Adding a new dependency

```bash
uv add <package>
uv add --dev <dev-package>
```

### Creating a new CLI command

Add to `pfun_cma_model/cli.py` using Click decorators.

### Running the full dev environment

```bash
uv run fastapi dev pfun_cma_model/app.py --port 8001
```

### Database operations

```bash
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "description"
```
