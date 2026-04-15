---
icon: lucide/rocket
---

# PFun CMA Model

Documentation for the `pfun-cma-model` package: a physiological chronometabolic model for glucose, hormone, and rhythm analysis.

## Table of Contents

- [API Reference](api.md)
- [Source repository](https://github.com/pfun-health/pfun-cma-model)
- [Project README](../README.md)

## Building documentation

Use Zensical to preview and build the site locally:

```bash
uv run zensical serve
uv run zensical build
```

## What is included

This documentation site includes:

- a home page for project overview and documentation guidance
- a dedicated API reference page with FastAPI/OpenAPI access instructions
- the core `pfun_cma_model` package reference and example usage

## API docs

The API is served by the FastAPI application in `pfun_cma_model/app.py`.
Run the app locally and inspect:

- `http://127.0.0.1:8001/docs`
- `http://127.0.0.1:8001/redoc`
- `http://127.0.0.1:8001/openapi.json`

For static reference, use the OpenAPI schema from the running app or generate a persisted schema file.
