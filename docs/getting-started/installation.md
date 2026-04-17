---
icon: lucide/download
---

# Installation

## Requirements

- **Python**: 3.12.11+ (< 3.13)
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- **OS**: Linux, macOS, Windows (WSL recommended for Windows)

## Install with uv (Recommended)

[`uv`](https://docs.astral.sh/uv/) provides ultra-fast dependency resolution and virtual environment management.

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via pip
pip install uv
```

### 2. Clone and set up

```bash
git clone https://github.com/pfun-health/pfun-cma-model.git
cd pfun-cma-model

# Create virtual environment & install dependencies
uv venv
uv sync
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

## Install with pip

```bash
pip install pfun-cma-model
```

!!! note "From source"
    For development, clone the repo and install in editable mode:
    ```bash
    pip install -e ".[dev]"
    ```

## Optional dependency groups

The project uses dependency groups for optional features:

| Group | Install Command | Description |
|-------|----------------|-------------|
| `dev` | `uv sync --group dev` | Development tools (ruff, mypy, pytest, zensical) |
| `ollama` | `uv sync --group ollama` | Ollama LLM backend |
| `google` | `uv sync --group google` | Google Gemini LLM backend |
| `perplexity` | `uv sync --group perplexity` | Perplexity LLM backend |
| `qt6` | `uv sync --group qt6` | Qt6 desktop GUI (PyQt6 + PySide6) |
| `datasette` | `uv sync --group datasette` | Interactive data exploration |

## Environment configuration

Copy the template and fill in your secrets:

```bash
cp .env.template .env
```

Key variables:

```ini
# LLM Backend (google | ollama | perplexity | openai)
LLM_BACKEND=ollama

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8001

# Redis (for rate limiting & caching)
REDIS_CONNECTION_STRING=redis://...
```

!!! tip "Dashlane CLI"
    If you have `dcli` (the Dashlane CLI) installed, secrets can be injected automatically:
    ```bash
    ./scripts/inject-secrets-env.sh
    ```

## Nix / devenv (Alternative)

If you use [Nix](https://nixos.org) with [devenv](https://devenv.sh):

```bash
# Enter the devenv shell (see flake.nix)
devenv shell

# — or with flakes directly —
nix develop --no-pure-eval
```

## Verify installation

```bash
# Check version
uv run pfun-cma-model version

# Run the test suite
uv run pytest -v

# Start the dev server
uv run fastapi dev pfun_cma_model/app.py --port 8001
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

**→ Next: [Quickstart Guide](quickstart.md)**
