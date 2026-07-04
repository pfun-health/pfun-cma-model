# CLEANROOM INSTRUCTIONS — TypeScript Functional Replica of `main` FastAPI API

## 1) Scope and objective

Build a TypeScript service that is **functionally equivalent** to the `origin/main` FastAPI API in this repository.

Parity reference snapshot: `origin/main` @ `<COMMIT_SHA>` (captured `<YYYY-MM-DD>`). Update this document whenever the parity target changes.

Compatibility target:
- Same externally reachable routes, route groups, and HTTP methods
- Same request/response shapes and media types
- Same default values, status codes, and key error behaviors
- Same websocket/socket.io interaction contract
- Same startup/shutdown side effects that affect API behavior

Do **not** reuse Python implementation code; implement a clean-room TypeScript equivalent.

---

## 2) Runtime/system behavior requirements

### 2.1 Application metadata
Expose OpenAPI docs and schema equivalent to FastAPI defaults:
- `/docs`
- `/redoc`
- `/openapi.json`

App metadata equivalents:
- Title: `PFun CMA Model Routing API`
- Description: `Server-side operations for operating the PFun CMA model; schema definitions, data IO, model execution.`
- Version format: `<package_version>-dev.<yyyymmddHHMMSS>` (timestamp-like dev suffix)

### 2.2 Startup behavior
On startup, replicate these observable behaviors:
1. Initialize template renderer for server-side HTML responses.
2. Attempt Redis connection (host/port/db/password from settings); do not crash if Redis is unavailable.
3. Download/ensure sample dataset availability.
4. Initialize admin DB models.

### 2.3 Shutdown behavior
On shutdown:
1. Dispose template instance.
2. Remove sample data.
3. Close Redis client if connected.

### 2.4 Static and template serving
- Mount static assets at `/static`.
- Serve HTML template responses for UI/demo routes listed below.

### 2.5 Middleware/security baseline
Replicate behavior-level protections:
- Security middleware with:
  - Rate limiting enabled (50 requests / 60 seconds)
  - IP auto-ban enabled (threshold 5, duration 3600 seconds)
  - Penetration detection enabled
  - Optional Redis-backed security storage
  - Blocked user agents: `badbot`, `evil-crawler`, `sqlmap`
  - Debug-query rejection behavior (`?debug=true` => HTTP 403 with `{"detail":"Debug mode not allowed"}`)
- Custom response headers (at minimum those documented in `docs/security.md`):
  - `Content-Security-Policy` (strict CSP with nonces), `Strict-Transport-Security` (HSTS)
  - `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`
  - `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy` (restricted feature access)
  - (Optional/legacy) `X-XSS-Protection: 1; mode=block` if present in `origin/main`
- CORS and trusted-host controls equivalent to debug/prod host allowlists in main branch config.
- Session middleware using configured secret key.
- Request-tracking middleware:
  - Build request metadata payload (IP, headers, path, method, query, cookies/session, timestamps)
  - If Redis present, store under `client_request:<ip>:<session_or_no-session>:<timestamp>` with 1h TTL
  - If Redis absent, continue request without failure

---

## 3) Route map (exact external contract)

## 3.1 Top-level routes

### `GET /health`
Response `200` JSON:
```json
{"status":"ok","message":"PFun CMA Model API is running."}
```

### `GET /health/ws/run-at-time`
- If socket.io session is active: `200` JSON
  ```json
  {"status":"ok","message":"'run-at-time' WebSocket is running."}
  ```
- Else: `503` JSON
  ```json
  {"status":"error","message":"'run-at-time' WebSocket is NOT running."}
  ```

### `GET /`
HTML template response (`index.html.jinja2` equivalent) with `year` and access message context.

### `GET /about`
HTML template response (`about-doc.html.jinja2`).

### `GET /pitch`
HTML template response (`pitch-doc.html.jinja2`).

### `GET /login`
HTML template response for admin login (`sqladmin/login.html` equivalent).

### `GET /favicon.ico`
Return ICO bytes with `image/x-icon`.

---

## 3.2 Model execution routes

### `POST /model/run`
Body: optional CMA model config object.
- If config provided: update model parameters.
- Execute model and return model dataframe JSON (`DataFrame.to_json()` style string).
Response:
- `200` with `Content-Type: application/json`
- Includes header `Access-Control-Allow-Origin: *`

### `POST /model/run-at-time`
Inputs: `t0`, `t1`, `n`, optional `config`.
- Runs point-in-time glucose computation over generated time vector.
- Returns JSON string output from `run_at_time_func`.
Error behavior:
- On failure: `500` JSON string with
  - `error: "failed to run at time. See error message on server log."`
  - `exception: <string>`

### `POST /model/run-at-time/stream`
Inputs: `t0`, `t1`, `n`, optional `config`.
Streaming response:
- `Content-Type: application/x-ndjson`
- Each line: `{"x":"<t>","y":"<Gt>"}` + newline
Error behavior during stream:
- Emit line with JSON:
  `{"error":"failed to run at time. See error message on server log.","status_code":500}`

### `POST /model/fit`
Inputs: `data` (`dict | string`), optional `config` (`string | CMAModelParams` in signature; effectively string path used).
Behavior:
- If data empty: load sample data.
- If data is string: parse JSON.
- Fit CMA model via optimizer and return `CMAFitResult.model_dump_json()` payload.
Responses:
- Success: `200` JSON
- Validation/decoding error: `400` JSON with fields `error`, `exception`, `exception_type`
- Other failure: `500` JSON with fields `error`, `exception`, `exception_type`

Compatibility quirk to preserve:
- Current implementation only initializes parsed config when `config` is string; other config forms can trigger runtime failure path. Replica should preserve externally observable behavior unless intentionally fixing with explicit product sign-off.

---

## 3.3 Parameters routes (`/params`)

### `GET /params/schema`
Returns JSON schema for CMA params (`CMAModelParams.model_json_schema()`), HTTP `200`.

### `GET /params/default`
Returns default CMA params (`CMAModelParams.model_dump_json()`), HTTP `200`.

### `POST /params/describe`
Body: params object.
Returns per bounded param (`d`, `taup`, `taug`, `B`, `Cm`, `toff`):
- `description`
- `qualitative`
- `value`
HTTP `200`.

### `POST /params/tabulate/{output_fmt}`
`output_fmt ∈ {json, html, md}`.
Body: params object.
Return table generated from bounded params:
- `md` -> `text/markdown`
- `html` -> `text/html`
- `json` -> JSON response containing serialized markdown table string

---

## 3.4 Data routes (`/data`)

Supported media types: `json | text | html | octet-stream`

Validation behavior:
- `nrows < -1` => `400` with detail: `nrows must be -1 (for full dataset) or a non-negative integer.`
- `pct0` must be in `[0.0, 1.0]`, else `400` with detail: `pct0 must be between 0.0 and 1.0.`

Selection behavior:
- Dataset source: sample data
- `row0 = int(pct0 * total_rows)`
- `nrows == -1`: return from `row0` to end
- `nrows >= 0`: wrap-around indexing from `row0` for `nrows` rows

### `GET /data/sample/download`
Defaults: `nrows=23`, `media_type=html`.
Outputs:
- `json` => `application/json` (records orient)
- `text` => `text/csv`
- `html` => `text/html`
- `octet-stream` => `501` with message: `Octet-stream download not implemented in non-streaming endpoint.`

### `GET /data/sample/stream`
Defaults: `pct0=0.5`, `nrows=10`, `media_type=text`.
Streaming outputs:
- `json` => `application/json`
- `text` => `text/csv`
- `html` => `text/html`
- `octet-stream` => `application/octet-stream`, chunked transfer, CSV body without header/index

---

## 3.5 LLM routes (`/llm`)

### `POST /llm/generate-scenario`
Inputs:
- `prompt: string`
- `include_sample_trace: bool` (default `false`)
- `include_recommendations: bool` (default `true`)

Behavior:
- Call async scenario generation
- Retry once on JSON serialization failure path
- Persist generated result to DuckDB via background task (`cma_recs` table; local/remote db path depends on debug setting)

Response:
- `200` with `Content-Type: application/json`
- Body is serialized `GeneratedScenario` object:
  - `forecasted_events: string`
  - `qualitative_description: string`
  - `parameters: { [name]: { value, description, stderr } }`
  - `recommendations: { [category]: string }` (when included)

### `POST /llm/generate-scenarios` (SSE)
Inputs:
- `prompts: string[]`
- `include_sample_trace: bool` (default `false`)
- `include_recommendations: bool` (default `true`)

Response class: `text/event-stream` equivalent.
Event contract:
- Event name: `generated_scenario`
- Event id: sequence starting at `"1"`
- Retry: `2300`

Compatibility quirk to preserve:
- Current code yields coroutine-produced data in SSE path (no await on internal call). Replica should preserve observed output behavior unless explicitly approved to fix.

---

## 3.6 Demo routes (`/demo`) — HTML pages

Return template-rendered HTML responses for:
- `GET /demo/llm` -> `llm-demo.html.jinja2`
- `GET /demo/data-stream` -> `data-stream-demo.html.jinja2`
- `GET /demo/run-at-time` -> `run-at-time-demo.html.jinja2`
- `GET /demo/canvas-wave` -> `canvas-wave-demo.html.jinja2`
- `GET /demo/full-model-run` -> `full-model-run-demo.html.jinja2`
- `GET /demo/webgl-demo` -> `webgl-demo.html.jinja2`

Context requirements:
- Include current `year`
- Include bounded parameter metadata used by interactive demos (name/value/description/min/max/step/default for each bounded key)
- Include CDN resource metadata with cache-busting query suffix when configured (`decache=true`)

---

## 3.7 Dexcom routes (`/dexcom`)

### `GET /dexcom/test`
`200` JSON: `{"message":"Dexcom router is working"}`

### `POST /dexcom/token`
Body includes `code`, `redirect_uri`.
- Exchanges auth code at Dexcom sandbox token endpoint
- Stores `dexcom_access_token` and `dexcom_refresh_token` in session
Errors:
- Missing code -> `400`
- Upstream non-200 -> propagate status with upstream JSON in detail

### `GET /dexcom/auth/callback`
- Requires `code` query param, else `400`
- Stores `dexcom_auth_code` in session
- Redirects to `/demo/dexcom`

### `GET /dexcom/users/self/egvs`
- Requires session access token (`401` if absent)
- Proxies Dexcom EGV endpoint with `startDate`/`endDate` query params

### `GET /dexcom/users/self/devices`
- Requires session access token (`401` if absent)
- Proxies Dexcom devices endpoint

---

## 3.8 Auth routes (`/auth`)

JWT behavior:
- Algorithm: `HS256`
- Secret from `JWT_SECRET_KEY` (fallback insecure default is present in Python implementation)
- Expiry minutes from `JWT_EXPIRATION_MINUTES` (default 1440)

Routes:
- `POST /auth/token/refresh` -> returns `{access_token, token_type, expires_in}`
- `POST /auth/token/verify` -> returns validity + identity + issued/expiry timestamps
- `GET /auth/user/me` -> returns `{id, first_name?, display_name?, picture?, provider}`
- `POST /auth/logout` -> confirmation payload
- `GET /auth/health` -> auth health payload
- `GET /auth/health/verify?token=...` -> token status payload (`no_token`, `valid`, `invalid`)

Auth enforcement:
- HTTP authorization header with JWT token is required on protected auth endpoints.
- Invalid token => `401` with `WWW-Authenticate: Bearer`

---

## 3.9 SSO routes (`/sso`)

Google SSO route set:
- `GET /sso/protected`
- `GET /sso/auth/login`
- `GET /sso/auth/logout`
- `GET /sso/auth/callback`

Behavior requirements:
- Initialize SSO backend in router lifespan.
- `/sso/auth/login` returns provider redirect.
- `/sso/auth/callback` verifies SSO response, creates JWT-like token using app secret/algorithm, stores in session + `token` cookie, then redirects to `/user/`.
- `/sso/protected` returns HTML greeting and immediate redirect meta tag toward `/admin/`.

---

## 4) Socket.IO/WebSocket contract

Mount socket.io ASGI endpoint at:
- `/socket.io/` for GET/POST polling and websocket route

Namespace behavior:
- Default namespace `/`
- Events:
  - `connect`
  - `disconnect`
  - `message` (echo-style response event `response`)
  - `run`
  - `run_full`

`run` event payload:
- Accept object or JSON string
- Fields: `t0` (default 0), `t1` (default 100), `n` (default 100), `config` (default `{}`)
- Stream emits `message` events with per-point JSON line from run-at-time stream (`{"x":"...","y":"..."}`)

`run_full` event payload:
- Same input defaults
- Stream emits `message` events with full model tuples (`{"t":"...","c":"...","m":"...","a":"..."}`)

Error behavior:
- On handler exception, emit `message` containing JSON error string.

---

## 5) CMA parameter and model contract requirements

Default CMA parameter model must match:
- `N=1024`
- `d=0.0`
- `taup=1.0`
- `taug=1.0`
- `B=0.05`
- `Cm=0.0`
- `toff=0.0`
- `tM=[7.0,11.0,17.5]`
- `seed=null`
- `eps=1e-18`

Bounded parameter keys and bounds:
- `d`: `[-12.0, 14.0]`
- `taup`: `[0.5, 3.0]`
- `taug`: `[0.1, 3.0]`
- `B`: `[0.0, 1.0]`
- `Cm`: `[0.0, 2.0]`
- `toff`: `[-3.0, 3.0]`

Model run output contract for full run (`run()`):
- Includes columns for `t, c, m, a, I_S, I_E, L`, meal components `g_0..g_n`, aggregate `G`, and `is_meal`.

---

## 6) Non-functional parity requirements

1. **Behavioral parity over refactor purity**: preserve externally observable outputs and status codes.
2. **Content-type parity**: keep all response media types matching current behavior.
3. **Error text parity**: maintain key error strings used by clients.
4. **Session/cookie semantics**: keep Dexcom and SSO flows session-backed.
5. **OpenAPI discoverability**: all HTTP routes above must appear in schema except explicitly hidden routes (e.g., favicon).

---

## 7) Explicit clean-room constraints

- Do not copy Python source logic verbatim.
- Reconstruct behavior from this requirement spec and independent implementation.
- Any intentional bug fixes that change API-observable behavior require explicit approval and a compatibility note.
