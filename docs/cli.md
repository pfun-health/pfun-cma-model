---
icon: lucide/terminal
---

# CLI Reference

The `pfun-cma-model` CLI provides commands for model fitting, scenario generation, parameter grid searches, and application launch.

## Usage

```bash
uv run pfun-cma-model [COMMAND] [OPTIONS]
```

## Commands

### `launch`

Launch the FastAPI application server.

```bash
uv run pfun-cma-model launch [OPTIONS] [EXTRA_ARGS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8001` | Port to listen on |
| `--reload` | off | Enable auto-reload for development |

Extra arguments are passed through to uvicorn:

```bash
# Launch with SSL
uv run pfun-cma-model launch \
  --ssl-certfile certs/example.crt \
  --ssl-keyfile certs/example.key
```

---

### `fit-model`

Fit the CMA model to glucose time series data.

```bash
uv run pfun-cma-model fit-model [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input-fpath` | `-i` | sample data | Path to input CSV file |
| `--output-dir` | `-o` | `results/` | Output directory |
| `--output-ftype` | `-T` | `png` | Figure format (`png` or `svg`) |
| `--N` | | `288` | Number of time points |
| `--plot` | | off | Show/save plot |
| `--model-config` | | `{}` | JSON string of model overrides |
| `--opts` | | | Curve-fit keyword arguments |

```bash
# Fit sample data with plot
uv run pfun-cma-model fit-model --plot

# Fit custom data, SVG output
uv run pfun-cma-model fit-model \
  -i data/patient_001.csv \
  -o results/ \
  -T svg \
  --N 1024 \
  --plot
```

---

### `generate-scenario`

Generate a physiologically valid scenario using the configured LLM backend.

```bash
uv run pfun-cma-model generate-scenario [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--query` | `"A healthy individual."` | Natural language description |

```bash
# Custom scenario
uv run pfun-cma-model generate-scenario \
  --query "a shift worker who sleeps from 2am to 10am"

# Default scenario
uv run pfun-cma-model generate-scenario
```

Results are saved to:

- `results/cma_recs.parquet` (Parquet)
- `results/duckdb-local.db` → table `cma_recs` (DuckDB)

---

### `run-param-grid`

Run a parameter grid search across the CMA model parameter space.

```bash
uv run pfun-cma-model run-param-grid [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `-N` | `-n` | `6` | Solution vector length (time points) |
| `-m` | | `3` | Parameter grid width (span) |
| `--params` | `-P` | all bounded | Parameters to include |

```bash
# Default 6×3 grid
uv run pfun-cma-model run-param-grid

# Larger grid with specific parameters
uv run pfun-cma-model run-param-grid -N 1024 -m 3 -P taug -P taup -P B
```

---

### `download-sample-data`

Download sample CGM data for testing.

```bash
uv run pfun-cma-model download-sample-data [--overwrite]
```

---

### `version`

Print the installed package version.

```bash
uv run pfun-cma-model version
# pfun-cma-model version: 0.4.196
```

---

### `run-doctests`

Run embedded doctests in the CLI module.

```bash
uv run pfun-cma-model run-doctests
```
