---
icon: lucide/book-open
---

# Jupyter Notebooks

## Available Notebooks

The `examples/notebooks/` directory contains interactive Jupyter notebooks for exploring the CMA model:

| Notebook | Description |
|----------|-------------|
| [`cma-model-modular-demos.ipynb`](https://github.com/pfun-health/pfun-cma-examples/blob/main/notebooks/cma-model-modular-demos.ipynb) | Comprehensive modular demo of all CMA model components |
| [`cma-model-tuning.ipynb`](https://github.com/pfun-health/pfun-cma-examples/blob/main/notebooks/cma-model-tuning.ipynb) | Parameter tuning and optimization walkthrough |
| [`visualize-generated-scenarios.ipynb`](https://github.com/pfun-health/pfun-cma-examples/blob/main/notebooks/visualize-generated-scenarios.ipynb) | Visualize LLM-generated scenario outputs |
| [`visualize-cma-parameter-grid.ipynb`](https://github.com/pfun-health/pfun-cma-examples/blob/main/notebooks/visualize-cma-parameter-grid.ipynb) | Explore precomputed parameter grids |
| [`kaggle-brist1d-fit-model-example.ipynb`](https://github.com/pfun-health/pfun-cma-examples/blob/main/notebooks/kaggle-brist1d-fit-model-example.ipynb) | Fit the CMA model to Kaggle BrisT1D dataset |
| [`plot-raw-glucose-data.ipynb`](https://github.com/pfun-health/pfun-cma-examples/blob/main/notebooks/plot-raw-glucose-data.ipynb) | Visualize raw CGM data from various sources |

## Running Notebooks

```bash
# Install Jupyter kernel
uv sync --group dev

# Launch Jupyter
uv run jupyter lab

# Or open a specific notebook
uv run jupyter lab examples/notebooks/cma-model-modular-demos.ipynb
```

## Rendered HTML Versions

Pre-rendered HTML versions of select notebooks are available for quick viewing:

- [cma-model-modular-demos.html](https://html-preview.github.io?url=https://github.com/pfun-health/pfun-cma-examples/blob/main/notebooks/cma-model-modular-demos.html)
- [cma-model-tuning.html](https://html-preview.github.io?url=https://github.com/pfun-health/pfun-cma-examples/blob/main/notebooks/cma-model-tuning.html)
- [visualize-generated-scenarios.html](https://html-preview.github.io?url=https://github.com/pfun-health/pfun-cma-examples/blob/main/notebooks/visualize-generated-scenarios.html)

## Code Samples

The `examples/code_samples/` directory contains standalone Python scripts:

| Script | Description |
|--------|-------------|
| `ollama_tool_calling_with_streaming.py` | Ollama tool calling with streaming responses |
| `fastapi_auth_oauth2_scopes.py` | FastAPI OAuth2 authentication with scopes |
| `sqladmin_authlib_auth_backend.py` | SQLAdmin with Authlib authentication |
| `sqladmin_basic_auth_backend.py` | SQLAdmin with basic authentication |
| `sqladmin_sso_auth_backend.py` | SQLAdmin with SSO authentication |

### Audio Processing

Experimental notebook for converting numerical arrays to audio tracks:

```
examples/code_samples/audio/
├── Numerical_Array_to_Audio_Track.ipynb
└── audacity_pipe_example.py
```
