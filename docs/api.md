---
icon: lucide/book-open
---

# API Reference

This project is built with [FastAPI](https://fastapi.tiangolo.com), and the API schema is generated automatically at runtime.

## Interactive API Docs

Start the server and access the built-in API documentation:

```bash
uv run fastapi dev pfun_cma_model/app.py --port 8001
```

| Viewer | URL | Description |
|--------|-----|-------------|
| **Swagger UI** | [`/docs`](http://localhost:8001/docs) | Interactive API explorer |
| **ReDoc** | [`/redoc`](http://localhost:8001/redoc) | Clean reference documentation |
| **OpenAPI JSON** | [`/openapi.json`](http://localhost:8001/openapi.json) | Raw schema for tooling |

## Route Groups

### Demo Routes (`/demo/`)

Interactive browser-based demonstrations:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/demo/llm` | LLM scenario generation UI |
| `GET` | `/demo/run-at-time` | WebSocket parameter-control demo |
| `GET` | `/demo/canvas-wave` | Canvas wave equation demo |
| `GET` | `/demo/full-model-run` | Full CMA model visualization |
| `GET` | `/demo/webgl-demo` | WebGL real-time plotting |
| `GET` | `/demo/data-stream` | Data streaming demo |

### LLM Routes (`/llm/`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate-scenario` | Generate a scenario from a query |

### Parameter Routes (`/params/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/params/` | Get current model parameters |
| `POST` | `/params/update` | Update model parameters |

### Data Routes (`/data/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/data/sample` | Get sample CGM data |
| `POST` | `/data/upload` | Upload CGM data |

### WebSocket Routes (`/ws/`)

| Protocol | Path | Description |
|----------|------|-------------|
| `WS` | `/ws/run` | Real-time model execution |

### Auth Routes (`/auth/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/auth/login` | OAuth2 login flow |
| `GET` | `/auth/callback` | OAuth2 callback |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check endpoint |
| `GET` | `/openapi.json` | OpenAPI schema |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

## Generating Static API Docs

Fetch the OpenAPI schema from the running app:

```bash
curl http://localhost:8001/openapi.json -o openapi.json
```

Then use tooling like [Redocly](https://redocly.com/) or [Swagger Codegen](https://swagger.io/tools/swagger-codegen/) to generate static reference pages.

## Client Generation

Generate typed API clients using the OpenAPI schema:

```bash
# Generate a TypeScript client
uv run openapi-generator-cli generate \
  -i openapi.json \
  -g typescript-fetch \
  -o generated_clients/typescript

# Generate a Python client
uv run openapi-generator-cli generate \
  -i openapi.json \
  -g python \
  -o generated_clients/python
```

!!! note
    The most accurate API documentation is always the live Swagger UI at `/docs`, since the schema is generated dynamically from the FastAPI app source.
