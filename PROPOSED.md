# PROPOSED: TypeScript Rewrite — Feature-Complete Clone of FastAPI API

## Goal

Produce a feature-complete, clean-room TypeScript re-implementation of the `pfun-cma-model` FastAPI backend (defined on `main` in `pfun_cma_model/api.py` and `pfun_cma_model/routes/*.py`) as a Hono/Node.js monorepo under `packages/`.

---

## Current State

The following was delivered in PR #97 (`feat: TypeScript clean-room implementation`):

### `@pfun/core` (`packages/core/`)

| Module | Status | Notes |
|---|---|---|
| `params.ts` | ✅ Complete | All 6 bounded params (`d`, `taup`, `taug`, `B`, `Cm`, `toff`) with bounds, defaults, steps, descriptions; Zod schema |
| `bounds.ts` | ✅ Complete | `Bounds` class with `clamp`, `normalize`, `denormalize` |
| `calc.ts` | ✅ Complete | `E`, `Light`, `exp`, `vectorized_G`, `meal_distr`, `linspace` |
| `model.ts` | ✅ Complete | `CMASleepWakeModel` with `run()`, `runAtTime()`, `runAtTimeStream()`, `runFullStream()` |
| `fit.ts` | ✅ Functional | Nelder-Mead-like optimizer over bounded params; simplified vs. SciPy `minimize` |
| Tests | ✅ 66 passing | `model.test.ts`, `params.test.ts`, `bounds.test.ts`, `calc.test.ts`, `fit.test.ts` |
| Benchmarks | ✅ Present | `packages/core/bench/` |

### `@pfun/api` (`packages/api/`)

| Route Group | Prefix | Status | Notes |
|---|---|---|---|
| Health / root | `/`, `/health`, `/about`, `/pitch` | ✅ Complete | Template rendering stub; favicon served from disk |
| Model execution | `/model/run`, `/model/run-at-time`, `/model/run-at-time/stream`, `/model/fit` | ✅ Complete | Mirrors `api.py` endpoints |
| Parameters | `/params/schema`, `/params/default`, `/params/describe`, `/params/tabulate/:fmt` | ✅ Complete | Mirrors `pfun_cma_model/routes/params.py` |
| Data | `/data/sample/download`, `/data/sample/stream` | ✅ Functional | See gap below re: sample data source |
| Auth | `/auth/token/refresh`, `/auth/token/verify`, `/auth/user/me`, `/auth/logout`, `/auth/health`, `/auth/health/verify` | ✅ Complete | JWT-based; mirrors `routes/auth.py` |
| SSO | `/sso/protected`, `/sso/auth/login`, `/sso/auth/logout`, `/sso/auth/callback` | ✅ Complete | Google OAuth2 PKCE flow; mirrors `routes/sso.py` |
| Dexcom | `/dexcom/test`, `/dexcom/token`, `/dexcom/auth/callback`, `/dexcom/users/self/egvs`, `/dexcom/users/self/devices` | ✅ Complete | Mirrors `routes/dexcom.py` |
| LLM | `/llm/generate-scenario`, `/llm/generate-scenarios` (SSE) | ⚠️ Stub | See gap below |
| Demo | `/demo/llm`, `/demo/data-stream`, `/demo/run-at-time`, `/demo/canvas-wave`, `/demo/full-model-run`, `/demo/webgl-demo` | ✅ Complete | Template-rendered demo pages |
| WebSocket health | `/health/ws/run-at-time` | ✅ Complete | Reports Socket.IO liveness |
| Socket.IO | `run`, `run_full`, `message` events | ✅ Complete | `packages/api/src/socketio.ts` |
| Security middleware | Rate limiting, IP ban, security headers, user-agent filter | ✅ Complete | In-memory; Redis-backed path stubbed |
| Tests | 51 passing | ✅ Passing | `health.test.ts`, `model-params.test.ts`, `routes.test.ts` |

---

## Remaining Work (Gaps)

The items below are required to reach full feature parity with the Python/FastAPI implementation.

### 1. LLM Backend Integration

**Python reference:** `pfun_cma_model/llm.py`, `pfun_cma_model/routes/llm.py`

The Python implementation dynamically loads one of four LLM backends (`ollama`, `openai`, `google`, `perplexity`) via `pfun_llm.backend.<name>` and calls `generate_scenario()` which returns a `GeneratedScenario` pydantic model populated by a real model inference call.

The TypeScript `generateScenario()` function in `packages/api/src/routes/llm.ts` is a hardcoded stub that returns static placeholder data.

**Required:**
- Implement an LLM backend abstraction in `@pfun/core` or `@pfun/api` with at minimum `ollama` support (local inference, no external API key required) and optionally `openai`/`google`.
- Connect `POST /llm/generate-scenario` and `POST /llm/generate-scenarios` to the real backend.
- Sanitize and validate the `prompt` field (Python uses `shlex.quote`; apply equivalent sanitization).
- Implement retry logic on JSON parse failure (already present structurally, but the retry currently calls `c.req.json()` a second time rather than retrying the generation).

### 2. Persistent Results Storage

**Python reference:** `pfun_cma_model/db.py` (`save2duckdb`), called as a background task in `routes/llm.py`

The Python API saves each successful LLM scenario generation to a DuckDB database (`results/duckdb-local.db` or `results/duckdb-remote.db`) in a background task.

**Required:**
- Add a lightweight persistence layer (DuckDB via `duckdb-node` or a JSON-lines append file as a fallback) to record LLM generation results.
- Wire it into `POST /llm/generate-scenario` as a fire-and-forget background task, mirroring `background_tasks.add_task(save2duckdb, ...)` in Python.

### 3. Admin Panel

**Python reference:** `pfun_cma_model/admin/` (models, views, auth, core, sso), mounted via `sqladmin` at `/admin/`

The Python API includes a full `sqladmin`-backed admin panel with:
- `User` model (id, name, email, is_admin, site_id, age, bio, hashed_password)
- `Site` model (id, name, users)
- SQLAlchemy async SQLite backend (`admin/core.py`)
- Alembic migrations for `users` and `sites` tables
- CRUD views (`UserAdmin`, `ReportView`)
- Auth backend integrated with JWT/SSO

The TypeScript implementation has no admin panel. The `/login` route renders a stub template and `/sso/protected` redirects to `/admin/`, but `/admin/` does not exist.

**Required:**
- Choose a Node.js admin framework (e.g., `adminjs`, `retool`, or a hand-rolled CRUD UI) or implement a minimal admin REST API with protected endpoints.
- Define `User` and `Site` entities (SQLite via `better-sqlite3` or `drizzle-orm` to mirror the Python SQLAlchemy schema).
- Implement password hashing (e.g., `bcrypt`) for the `hashed_password` field (noted as TODO in the Python implementation as well).
- Mount admin routes under `/admin/` and protect them with the existing JWT auth middleware.
- Add database migrations (e.g., `drizzle-kit` or raw SQL scripts).

### 4. Real Sample Data Source for `/data` Routes

**Python reference:** `pfun_cma_model/data.py` (`read_sample_data`)

The Python implementation reads a real CSV dataset from disk (`pfun_cma_model/static/` or a configured data path). The TypeScript implementation generates synthetic sample data by running `CMASleepWakeModel().run()` and only returns four fields (`t`, `G`, `c`, `m`).

**Required:**
- Bundle the sample CSV dataset into `packages/api/` (copy from Python static assets or reference a shared path).
- Implement `readSampleData()` in `packages/api/src/data.ts` to parse and return the real dataset.
- Replace the synthetic `getSampleData()` call in `routes/data.ts` with the real dataset.
- Ensure the full column set is returned (not just `t`, `G`, `c`, `m`).

### 5. Template Rendering

**Python reference:** `pfun_cma_model/templates/` (Jinja2), `pfun_cma_model/misc/templating.py`

The TypeScript implementation uses a minimal stub template renderer that returns a generic HTML skeleton regardless of which template is requested. The actual Jinja2 templates (index, about-doc, pitch-doc, demo pages, etc.) are not rendered.

**Required:**
- Copy or symlink the existing Jinja2 templates from `pfun_cma_model/templates/` into `packages/api/templates/`.
- Integrate Nunjucks (already a dependency) as the template engine, replacing the current stub `createTemplateRenderer`.
- Wire template data (e.g., model params, CDN resources, year) into each demo route context, mirroring `routes/demo.py`.
- Serve `pfun_cma_model/static/` from `packages/api/` (configure `staticDir` to point to the shared static directory or copy assets).

### 6. Security Middleware Parity

**Python reference:** `pfun_cma_model/security.py`, `guard.SecurityMiddleware`

The Python implementation uses `fastapi-guard` (`SecurityMiddleware`) which provides IP-info-based blocking, configurable threat detection, and a richer penetration-detection ruleset.

**Required:**
- Expand the TypeScript `debugQueryRejection` middleware to block the same SQL injection / path traversal patterns that `fastapi-guard` covers.
- Add the `X-Content-Type-Options: nosniff` header (currently missing; Python adds it implicitly via `fastapi-guard`).
- Implement Redis-backed rate limiting (the `ioredis` dependency is already present; the in-memory fallback can remain for development).
- Add `TrustedHostMiddleware` equivalent: reject requests whose `Host` header is not in `config.trustedHosts`.

### 7. Params Describe / Tabulate — Full Parity

**Python reference:** `pfun_cma_model/engine/cma_model_params.py` (`describe`, `generate_qualitative_descriptor`, `generate_markdown_table`)

The TypeScript `/params/describe` and `/params/tabulate/:fmt` endpoints call methods that are not yet implemented in `@pfun/core/params.ts`. They return placeholder or partial output.

**Required:**
- Implement `describe(key)` on `CMAModelParams` (return the long-form description string for a bounded param key).
- Implement `generateQualitativeDescriptor(key)` (return a qualitative health interpretation for a param value, as done in Python).
- Implement `generateMarkdownTable(outputFmt)` supporting `md`, `html`, and `json` formats.

### 8. Lifespan / Startup and Shutdown Hooks

**Python reference:** `pfun_cma_model/api.py` (`@asynccontextmanager async def lifespan`)

The Python app initializes the admin DB models and SSO backend on startup. The TypeScript app has no equivalent lifecycle management.

**Required:**
- Add a startup hook in `packages/api/src/index.ts` that initializes the database schema (run migrations / `CREATE TABLE IF NOT EXISTS`) and any other resources.
- Add a graceful shutdown hook to close database connections and shut down Socket.IO cleanly.

---

## Out of Scope for This Branch

The following Python features exist in `main` but are explicitly **not** required for feature parity in this rewrite (they are noted as incomplete or future work in `TODO.md`):

- `pfun_common` package integration (settings, logging utilities) — the TypeScript implementation uses `config.ts` and `console.*` equivalents.
- `pfun_llm` multi-backend plugin architecture — a single-backend stub is acceptable for initial feature parity.
- OpenRLHF / Ray-based RL training pipeline.
- Cloudflare Worker / Pulumi infra.
- Nightscout integration.
- Kaggle dataset / HuggingFace `IterableDataset` creation.

---

## Acceptance Criteria

The typescript-rewrite branch is considered feature-complete when:

1. All route groups in the table above show ✅ Complete (no ⚠️ stubs).
2. `pnpm test` passes in all packages with ≥ 90% route coverage.
3. The Hono app can be started with `pnpm start` and passes the same smoke tests as `uvicorn pfun_cma_model.api:app`.
4. The security headers contract defined in `docs/security.md` is satisfied (Content-Security-Policy, Strict-Transport-Security, X-Frame-Options, Referrer-Policy, Permissions-Policy, X-Content-Type-Options).
5. `/admin/` is accessible and protected by JWT auth.
6. LLM endpoints return real (non-stub) output when a backend (e.g., local Ollama) is configured.
7. `Dockerfile.node` builds and produces a runnable image.
