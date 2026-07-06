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

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (with Compose v2)
- Ports **8000** (API) and **6379** (Redis) free on the host

### Quick start

```bash
# Build and start all services in the background
docker compose up -d

# Check logs
docker compose logs -f

# Stop everything
docker compose down
```

### Full rebuild (clean)

```bash
docker compose up -d \
  --build \
  --renew-anon-volumes \
  --remove-orphans
```

### Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `api`   | Custom (`Dockerfile.node`) | `8000` | Hono TypeScript HTTP API |
| `redis` | `redis:7-alpine` | `6379` | Cache & rate-limiting backend |

### Dockerfile

The production `Dockerfile.node` uses a multi-stage build:

```dockerfile
# Deps stage — install all dependencies
FROM node:22-alpine AS deps
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY packages/core/package.json packages/core/
COPY packages/api/package.json packages/api/
RUN pnpm install --frozen-lockfile --prod=false

# Build stage — compile TypeScript
FROM deps AS build
COPY tsconfig.base.json ./
COPY packages/core/ packages/core/
COPY packages/api/ packages/api/
RUN pnpm -r build

# Production stage — only production deps + compiled output
FROM node:22-alpine AS production
ENV NODE_ENV=production
COPY --from=build /app/packages/core/dist packages/core/dist
COPY --from=build /app/packages/api/dist packages/api/dist
CMD ["node", "packages/api/dist/index.js"]
```

### Configuration

All environment variables are read from `.env` at startup. See `.env.template` for the full reference.

Key variables for Docker Compose:

| Variable | Default (`.env`) | Description |
|----------|-------------------|-------------|
| `PORT` | `8000` | API listen port |
| `HOST` | `0.0.0.0` | Bind address |
| `REDIS_HOST` | `redis` | Redis hostname (matches compose service name) |
| `REDIS_PORT` | `6379` | Redis port |
| `JWT_SECRET_KEY` | `dev-secret-...` | Change for production! |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |

### Optional: LLM backend (Ollama)

1. Uncomment the `ollama` service block in `docker-compose.yml`
2. Set `LLM_BACKEND=ollama` in `.env`
3. Start the service and pull a model:

```bash
docker compose --profile llm up -d ollama
docker compose exec ollama ollama pull llama3.2
```

> **GPU acceleration**: The commented block includes an NVIDIA device reservation. Remove the `deploy.resources` section if you don't have an NVIDIA GPU + `nvidia-container-toolkit` installed.

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
