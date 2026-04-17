---
icon: lucide/git-pull-request
---

# Contributing

## Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/pfun-health/pfun-cma-model.git
cd pfun-cma-model

# 2. Install all dependencies including dev tools
uv venv && uv sync --group dev

# 3. Activate the environment
source .venv/bin/activate

# 4. Verify everything works
uv run pytest -v
```

## Code Quality

### Linting

```bash
# Check for issues
uv run ruff check .

# Auto-fix
uv run ruff check --fix .
```

### Formatting

```bash
uv run ruff format .
```

### Type Checking

```bash
# Full project
uv run mypy pfun_cma_model

# Specific file
uv run mypy pfun_cma_model/engine/fit.py
```

## Testing

```bash
# Run all tests
uv run pytest

# Verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_fit.py

# Run specific test function
uv run pytest tests/test_fit.py::test_estimate_mealtimes_invalid_input

# Pattern matching
uv run pytest -k "test_estimate"

# With coverage
uv run pytest --cov=pfun_cma_model
```

### Test Guidelines

- Place tests in `tests/` directory
- Name test files: `test_<module>.py`
- Use descriptive names: `test_<function>_raises_on_invalid_input`
- Use `pytest.raises` for exception testing
- Use fixtures for common test data

```python
import pytest
import pandas as pd

def test_estimate_mealtimes_invalid_input():
    with pytest.raises(ValueError, match="Input data cannot be None or empty."):
        estimate_mealtimes(None)
```

## Code Style

### Imports

```python
# Group: stdlib → third-party → local
import os
from typing import Optional

import pandas as pd
from fastapi import APIRouter

from pfun_cma_model.engine.fit import estimate_mealtimes
```

### Type Hints

Always use type hints:

```python
def process_data(data: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    ...
```

### Docstrings

Use Google-style docstrings:

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

## Documentation

### Building docs locally

```bash
# Install zensical
uv sync --group dev

# Preview with live reload
uv run zensical serve

# Build static site
uv run zensical build
```

### Documentation structure

```
docs/
├── index.md                    # Home page
├── getting-started/
│   ├── installation.md         # Install guide
│   └── quickstart.md           # Quick start
├── model/
│   ├── overview.md             # CMA model overview
│   ├── parameters.md           # Parameter reference
│   └── fitting.md              # Fitting pipeline
├── demos/
│   ├── gallery.md              # Visual gallery
│   ├── llm-scenarios.md        # LLM generation
│   ├── websocket-streaming.md  # WebSocket demos
│   ├── qt-gui.md               # Desktop GUI
│   └── notebooks.md            # Jupyter notebooks
├── cli.md                      # CLI reference
├── api.md                      # API reference
├── deployment.md               # Deployment guide
├── security.md                 # Security config
└── contributing.md             # This file
```

## Adding Dependencies

```bash
# Runtime dependency
uv add <package>

# Dev dependency
uv add --dev <package>

# Dependency group
uv add --group ollama ollama
```

## Database Migrations

```bash
# Apply migrations
uv run alembic upgrade head

# Create migration
uv run alembic revision --autogenerate -m "description"
```
