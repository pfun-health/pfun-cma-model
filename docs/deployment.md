---
icon: lucide/cloud
---

# Deployment

## Local Development

```bash
# Start the dev server with auto-reload
uv run fastapi dev pfun_cma_model/app.py --port 8001

# Or via CLI with SSL
uv run pfun-cma-model launch --port 8001 --reload \
  --ssl-certfile certs/example.crt \
  --ssl-keyfile certs/example.key
```

## Docker

### Build and run with Docker Compose

```bash
# Full rebuild with clean volumes
docker compose up -d \
  --build \
  --renew-anon-volumes \
  --remove-orphans

# Or use the convenience script
./scripts/full-rebuild.sh
```

### Dockerfile

The production `Dockerfile` uses a multi-stage build:

```dockerfile
# Build stage
FROM python:3.12-slim AS builder
RUN pip install uv
COPY . /app
WORKDIR /app
RUN uv sync --frozen

# Runtime stage
FROM python:3.12-slim
COPY --from=builder /app /app
WORKDIR /app
CMD ["uv", "run", "gunicorn", "pfun_cma_model.app:app"]
```

## Kubernetes (Helm)

### Convert docker-compose to Helm Chart

```bash
# Generate Helm chart from docker-compose.yml
kompose convert -c -o pfun-cma-model-chart

# Build the chart package
helm package pfun-cma-model-chart --destination dist/

# Install
helm install pfun-cma-model pfun-cma-model-chart/
```

## Google Cloud Platform (Cloud Run)

### Publish a new version

```bash
./scripts/new-version.sh
```

This script:

1. Builds the Docker image
2. Pushes to Google Container Registry
3. Deploys a new Cloud Run revision

### Cloud Build

The `cloudbuild.yml` defines the CI/CD pipeline for automated builds on push.

## Nix Images

The repository now defines two Nix build outputs for deployment artifacts:

```bash
nix build .#oci-image
nix build .#vm-image
```

- `.#oci-image` builds an OCI container archive for registry publishing.
- `.#vm-image` builds a qcow-based VM image for VM deployments.

The GitHub Actions workflow at `.github/workflows/docker-push.yaml` builds and publishes the OCI image to GHCR and uploads the VM image artifact when a GitHub Release is published.

## Domain Configuration

| Service | Domain | Description |
|---------|--------|-------------|
| Landing page | `pfun.me` | Public homepage |
| Demo frontend | `pfun.app` | Interactive demos |
| Backend API | `api.pfun.run` | Production API |
| Dev/staging | `cloud.tail38611b.ts.net` | Tailscale private network |

## Database Migrations

```bash
# Run pending migrations
uv run alembic upgrade head

# Create a new migration
uv run alembic revision --autogenerate -m "add new table"

# Rollback one step
uv run alembic downgrade -1
```

## Environment Variables

See `.env.template` for all required configuration:

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_BACKEND` | ✅ | LLM provider (`ollama`, `google`, etc.) |
| `SERVER_HOST` | ✅ | Bind host |
| `SERVER_PORT` | ✅ | Bind port |
| `REDIS_CONNECTION_STRING` | | Redis URL for caching/rate limiting |
| `DATABASE_URL` | | SQLite/PostgreSQL connection |
| `GOOGLE_CLIENT_ID` | | OAuth2 client ID |
| `GOOGLE_CLIENT_SECRET` | | OAuth2 secret |
