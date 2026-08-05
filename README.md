# PFun CMA Model

**Circadian Metabolic Analysis** — a clean-room TypeScript implementation of sleep-wake models that simulate the human circadian rhythm and its effects on metabolic processes.

This monorepo provides a model engine (`@pfun/core`), an HTTP API server (`@pfun/api`), and a command-line interface (`cli`) for running, fitting, and serving CMA sleep-wake simulations.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage — `@pfun/core`](#usage--pfunkeywordscore)
- [Usage — `@pfun/api`](#usage--pfunapi)
- [Usage — CLI](#usage--cli)
- [Development](#development)
- [Testing](#testing)
- [Architecture Notes](#architecture-notes)
- [License](#license)

---

## Features

- **Circadian Sleep-Wake Modeling** — Simulates cortisol (C), melatonin (M), adiponectin (A), light exposure (L), and glucose (G) dynamics over a 24-hour cycle.
- **Dual Implementation Architecture** — A low-level engine (Implementation B) using `Float64Array` for performance, alongside a higher-level API (Implementation A) with ergonomic array operations.
- **Parameter Grid Search** — Explore the parameter space with `PFunCMAParamsGrid` over configurable ranges.
- **Model Fitting** — Coordinate-descent optimization to fit model parameters to observed glucose data.
- **Scenario Generation** — Keyword-driven generation of realistic metabolic scenarios (night owl, early bird, diabetic profiles).
- **Zod-Validated Parameters** — All model parameters are validated and typed via Zod schemas.
- **REST API** — Hono-based HTTP server with WebSocket support (Socket.IO), security middleware, JWT auth, SSO, and Dexcom integration.
- **CLI** — Commander-based CLI for launching the API, running parameter grids, fitting models, and generating scenarios.
- **Fully Tested** — 23+ tests across core, API, and CLI packages using Vitest.

---

## Prerequisites

- **Node.js** >= 20 (ESM modules, ES2022 target)
- **pnpm** >= 11.3.0 (see `devEngines` in root `package.json`)

---

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd pfun-cma-model

# Install all workspace dependencies
pnpm install

# Build all packages (core → api → cli)
pnpm build

# Run all tests
pnpm test
```

### Quick verification

```bash
# Run the core model test suite
pnpm --filter @pfun/core test

# Launch the API server (after building)
pnpm --filter @pfun/api dev
```

---

## Project Structure

```
pfun-cma-model/
├── package.json              # Workspace root with shared devDependencies
├── pnpm-workspace.yaml       # Declares packages/* as workspace members
├── tsconfig.base.json        # Shared TypeScript config (ES2022, bundler resolution)
├── vitest.config.ts          # Root Vitest config (globals, node environment)
├── packages/
│   ├── core/                 # @pfun/core — Model engine library
│   │   ├── src/
│   │   │   ├── index.ts      # Public API exports
│   │   │   ├── cma.ts        # CMASleepWakeModel (primary, Impl B)
│   │   │   ├── engine.ts     # Low-level engine (Float64Array-based)
│   │   │   ├── cma_model_params.ts  # Zod schema for Impl B params (N=24 default)
│   │   │   ├── grid.ts       # PFunCMAParamsGrid
│   │   │   ├── llm.ts        # generateScenario (keyword-based)
│   │   │   ├── model.ts      # CMASleepWakeModel (secondary, Impl A)
│   │   │   ├── params.ts     # Param metadata + Zod schema (Impl A, N=1024 default)
│   │   │   ├── calc.ts       # Higher-level math functions
│   │   │   ├── fit.ts        # Model fitting (coordinate descent)
│   │   │   ├── bounds.ts     # Bounds constraint class
│   │   │   └── __tests__/    # Tests for Implementation B exports
│   │   ├── tests/            # Tests for Implementation A exports
│   │   ├── dist/             # Built ESM output (keep in sync with source)
│   │   ├── tsup.config.ts    # Bundle config (ESM, DTS, sourcemaps)
│   │   └── vitest.config.ts  # Package-level test config
│   ├── api/                  # @pfun/api — Hono HTTP API server
│   │   ├── src/
│   │   │   ├── index.ts      # createApp(), main entry point
│   │   │   ├── app.ts        # Re-exported Hono app instance
│   │   │   ├── config.ts     # Environment-based configuration
│   │   │   ├── socketio.ts   # Socket.IO WebSocket handler
│   │   │   ├── routes/       # Route groups (health, model, params, data, auth, etc.)
│   │   │   ├── middleware/   # Security middleware (CORS, rate limiting, etc.)
│   │   │   └── __tests__/    # API tests
│   │   ├── tests/            # Integration tests
│   │   └── tsup.config.ts
│   └── cli/                  # CLI — Commander-based command-line tool
│       ├── src/
│       │   ├── index.ts      # CLI entry point with Commander commands
│       │   └── __tests__/    # CLI tests
│       └── tsconfig.json
└── docs/
    ├── api.md                # API documentation
    └── deployment.md         # Deployment guide
```

### Package relationships

```
@pfun/core     ←  Runtime dependency: zod
     ↑
@pfun/api      ←  Depends on @pfun/core + hono, ioredis, socket.io, etc.
     ↑
cli            ←  Depends on @pfun/core + @pfun/api + commander + zod@4
```

---

## Usage — `@pfun/core`

The core library provides two parallel implementations. The primary exports (Implementation B) are the default API for external consumers.

### Importing

```typescript
import {
  // Implementation B (primary)
  CMASleepWakeModel,
  CMAModelParamsSchema,
  PFunCMAParamsGrid,
  generateScenario,
  runCMAModel,
  exp_clipped,
  expit,
  Light_pfun,
  E_pfun,
  K_pfun,

  // Implementation A (secondary)
  CMASleepWakeModel as ModelA,  // Note: same class name, different module
  getDefaultParams,
  getParamsJsonSchema,
  Bounds,
  fitModel,
  exp,
  expitPfun,
  E_norm,
  Light,
  E,
  K,
  K_vec,
  computeG,
  linspace,
  normalize,
} from '@pfun/core';
```

### Creating a model and running a simulation

**Implementation B (primary)** — uses `CMASleepWakeModel` from `cma.ts` with `solve()`:

```typescript
import { CMASleepWakeModel } from '@pfun/core';

// Create model with default parameters (N=24, 3 default meals)
const model = new CMASleepWakeModel();

// Run the simulation
model.solve();

// Access the solution
const solution = model.solution!;
console.log(solution.t);     // Time points (length N)
console.log(solution.L);     // Light exposure
console.log(solution.M);     // Melatonin
console.log(solution.C);     // Cortisol
console.log(solution.A);     // Adiponectin
console.log(solution.I_S);   // Sleep pressure (insulin sensitivity proxy)
console.log(solution.I_E);   // Extracellular insulin
console.log(solution.G);     // Glucose response
```

**Implementation A (secondary)** — uses `CMASleepWakeModel` from `model.ts` with `run()`:

```typescript
import { CMASleepWakeModel } from '@pfun/core';

const model = new CMASleepWakeModel({ N: 100 });
const results = model.run();

// Each row contains: t, c, m, a, I_S, I_E, L, G, g_0..g_n, is_meal
console.log(results[0].t);    // Time in hours
console.log(results[0].c);    // Cortisol
console.log(results[0].G);    // Glucose
console.log(results[0].is_meal); // Boolean, true near meal times
```

### Custom parameters

```typescript
// Override specific parameters at construction
const model = new CMASleepWakeModel({
  N: 288,           // 5-minute intervals
  d: 2.0,           // Time zone offset (hours)
  taup: 1.5,        // Photoperiod duration
  taug: 1.2,        // Glucose response time constant
  B: 0.1,           // Glucose baseline
  Cm: 0.5,          // Cortisol sensitivity
  toff: 1.0,        // Solar noon offset
  tM: [8.0, 12.0, 18.0],  // Meal times
  seed: 42,         // Reproducible randomness
});

// Update parameters after construction (Impl B)
model.update({ d: -1.0, taup: 0.8 });
model.solve();

// Update parameters after construction (Impl A)
model.updateParams({ B: 0.2 });
const results = model.run();
```

### Parameter grid search

```typescript
import { PFunCMAParamsGrid } from '@pfun/core';

const grid = new PFunCMAParamsGrid({
  N: 24,                      // Time points per simulation
  m: 3,                       // Grid points per parameter
  keys: ['taug', 'taup', 'B', 'Cm'],  // Parameters to vary
});

grid.run();

// Each entry in collection contains params + solution data
for (const entry of grid.collection) {
  console.log(entry.taug, entry.taup, entry.B, entry.Cm);
}
```

### Model fitting

```typescript
import { fitModel } from '@pfun/core';

const data = {
  t: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
  G: [5.0, 4.8, 4.9, 5.2, 6.5, 7.0, 5.5, 5.0, 6.0, 6.8, 5.2, 4.9],
};

const result = fitModel(data, { N: 24 }, 200, 1e-6);

console.log(result.params);   // Fitted parameters
console.log(result.residual); // Sum of squared errors
console.log(result.success);  // Whether optimization converged
```

### Calc functions (Implementation A)

```typescript
import { exp, expitPfun, E, K, Light, linspace, computeG, normalize } from '@pfun/core';

// Safe exponential (clipped to avoid overflow)
exp(1000);   // ≈ Math.exp(709)

// Sigmoid / expit variants
expitPfun(0);    // 0.5
E(0);            // 0.5
E_norm(0);       // 0.0 (normalized to [-1, 1])

// Glucose response kernel
K(0.5);   // > 0 (peaks near 0.5)

// Light function
Light(0);   // 1.0 (max at noon)

// Generate time vector
const t = linspace(0, 24, 97);  // 97 points from 0 to 24

// Full glucose computation
const I_E = new Array(97).fill(0.05);
const gMeals = computeG(t, I_E, [7.0, 12.0, 18.0], 1.0, 0.05, 0.0, 0.0, true);

// Normalize data
normalize([0, 5, 10]);           // [0, 0.5, 1]
normalize([0, 10], -1, 1);      // [-1, 1]
```

### Engine functions (Implementation B)

```typescript
import { runCMAModel, exp_clipped, expit, Light_pfun, E_pfun, K_pfun } from '@pfun/core';

// Low-level model execution with Float64Array
const N = 24;
const t = new Float64Array([/* time points */]);
const tM = new Float64Array([7.0, 11.0, 17.5]);
const seed = new Int32Array([42]);
const outL = new Float64Array(N);
const outM = new Float64Array(N);
const outC = new Float64Array(N);
const outA = new Float64Array(N);
const outIS = new Float64Array(N);
const outIE = new Float64Array(N);
const outG = new Float64Array(N);
const outGComp = new Float64Array(N * tM.length);

runCMAModel(
  t, N, 0.0, 1.0, 1.0, null, 0.05, 0.0, 0.0,
  tM, tM.length, seed, 1e-18,
  outL, outM, outC, outA, outIS, outIE, outG, outGComp,
);
```

### Scenario generation

```typescript
import { generateScenario } from '@pfun/core';

// Standard profile
const standard = generateScenario('A healthy individual');

// Night owl
const nightOwl = generateScenario('night owl');
// → toff = 2.5

// Early bird
const earlyBird = generateScenario('early bird');
// → toff = -2.0

// Diabetic profile
const diabetic = generateScenario('person has diabetes');
// → B = 0.2, taug = 2.5
```

---

## Usage — `@pfun/api`

### Quick start

```typescript
import { createApp } from '@pfun/api';

const { app, config } = createApp();

// app is a Hono instance ready to serve
// Default: listens on http://0.0.0.0:8000
```

Or run the server directly:

```bash
# After building
node packages/api/dist/index.js

# Or with tsx during development
pnpm --filter @pfun/api dev
```

### Configuration

The API is configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `DEBUG` | `false` | Enable debug mode |
| `REDIS_URL` | `null` | Redis connection URL |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `JWT_SECRET_KEY` | `insecure-default-secret` | JWT signing key |
| `CORS_ORIGINS` | `*` | CORS allowed origins (comma-separated) |
| `DEXCOM_CLIENT_ID` | — | Dexcom API client ID |
| `GOOGLE_CLIENT_ID` | — | Google SSO client ID |

### Endpoints overview

| Route | Description |
|---|---|
| `GET /` | Landing page (HTML) |
| `GET /about` | About page |
| `GET /login` | Login page |
| `GET /health` | Health check (`{ status: "ok" }`) |
| `GET /health/ws/run-at-time` | WebSocket health status |
| `GET /openapi.json` | OpenAPI 3.1 schema |
| `GET /docs` | Swagger UI documentation |
| `GET /redoc` | ReDoc documentation |
| `POST /model/run` | Run model simulation |
| `GET /params` | List model parameters |
| `POST /auth/login` | JWT authentication |
| `POST /auth/register` | User registration |
| `GET /sso/google` | Google SSO login |
| `GET /dexcom/auth` | Dexcom OAuth flow |
| `POST /llm/generate` | LLM-powered scenario generation |
| `GET /demo` | Interactive demo (HTML) |

### WebSocket (Socket.IO)

The API supports real-time model streaming via Socket.IO:

- **`run`** — Stream run-at-time results (emits `"message"` events with `{x, y}` points)
- **`run_full`** — Stream full model results (emits `"message"` events with `{t, c, m, a}` tuples)

```javascript
// Client-side example
const socket = io('http://localhost:8000');

socket.emit('run', {
  t0: 0,
  t1: 24,
  n: 100,
  config: { B: 0.08 }
});

socket.on('message', (data) => {
  console.log(JSON.parse(data)); // { x: "12.0", y: "5.2" }
});
```

### API programmatic usage

```typescript
import { createApp } from '@pfun/api';
import { serve } from '@hono/node-server';

const { app, config } = createApp();

serve({
  fetch: app.fetch,
  port: config.port,
  hostname: config.host,
}, (info) => {
  console.log(`API listening on http://${config.host}:${info.port}`);
});
```

---

## Usage — CLI

The `cli` package provides a Commander-based command-line interface for the
pfun-cma-model project. It exposes commands for launching the API server,
fitting the sleep-wake model, generating metabolic scenarios, running
parameter grid searches, downloading sample data, and benchmarking.

The CLI can be invoked either via `pnpm --filter cli start` (after building)
or, if the package binary is linked, directly as `pfun-cma-model`.

### Prerequisites

The CLI depends on both `@pfun/core` and `@pfun/api`. These must be built
before the CLI will function correctly:

```bash
pnpm --filter core build
pnpm --filter api build
pnpm --filter cli build
```

### Building the CLI

```bash
pnpm --filter cli build
```

This compiles the TypeScript source from `packages/cli/src/` to
`packages/cli/dist/` using the project's `tsconfig.json` (target ES2022).
The output is ESM-compatible (`"type": "module"` in package.json).

### Running the CLI

**Via pnpm (recommended during development):**

```bash
pnpm --filter cli start --help
pnpm --filter cli start <command> [options]
```

**Globally via the linked binary:**

If the package has been linked with `pnpm link --global` or installed from a
registry, the `pfun-cma-model` binary is available directly:

```bash
pfun-cma-model --help
pfun-cma-model <command> [options]
```

---

### Command Reference

#### `launch`

Start the HTTP API server.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host <host>` | string | `127.0.0.1` | Host address to bind the server to. |
| `--port <port>` | string | `8001` | Port number to listen on. |

**How it works:** The command dynamically imports the `@pfun/api` package's
compiled entry point (`../../api/dist/index.js` relative to the CLI dist
directory) at runtime. This avoids a hard compile-time dependency on the API
server — the API only needs to be present and built when `launch` is actually
invoked.

**Usage:**

```bash
pnpm --filter cli start launch --port 8080 --host 0.0.0.0
```

**Error handling:** If the dynamic import fails (e.g., `@pfun/api` has not
been built), the command prints a descriptive message and exits with code 1:

```
Failed to launch API server. Has it been built? Run: pnpm --filter api build
```

---

#### `fit-model`

Fit the CMA sleep-wake model to a dataset.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--N <number>` | integer (parsed) | `288` | Number of time points. Must be an integer >= 2. |

**How it works:** Parses and validates `--N`, then instantiates a
`CMASleepWakeModel` with `{ N: n }` and calls `.solve()`. The 288 default
corresponds to 5-minute intervals across a 24-hour period.

**Usage:**

```bash
pnpm --filter cli start fit-model --N 576
```

**Error handling:** If `--N` is not a valid integer or is less than 2, the
command prints `N must be an integer >= 2` to stderr and exits with code 1.

**Example output:**

```
Fitting model with N=288...
...wrote fitted model params to: fit_result.json
```

---

#### `generate-scenario`

Generate a realistic pfun (phenotype-function) metabolic scenario from a text
prompt.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--query <query>` | string | `A healthy individual.` | Natural-language description of the scenario. |

**How it works:** Calls the `generateScenario()` function from `@pfun/core`
with the provided query string, then prints the result as formatted JSON and
reports database persistence.

**Usage:**

```bash
pnpm --filter cli start generate-scenario --query "night owl with type 2 diabetes"
```

**Note:** The status message truncates the query to 20 characters for display:

```
Generating a scenario from prompt:
	'night owl with type 2...'
```

**Example output:**

```json
{
  "labels": ["Sleep", "Wake", "Exercise"],
  "values": [0.15, 0.7, 0.15],
  "description": "A healthy individual."
}
```

---

#### `run-param-grid`

Run a parameter grid search over the PFun CMA model solution space.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-N, --N <number>` | integer (parsed) | `6` | Length of the solutions vector. Must be >= 2. |
| `-m, --m <number>` | integer (parsed) | `3` | Grid width (granularity). Must be >= 2. |

**How it works:** Creates a `PFunCMAParamsGrid` instance with `{ N, m }` and
calls `.run()` to execute the grid search. Both values are validated as
integers >= 2 before construction.

**Usage:**

```bash
pnpm --filter cli start run-param-grid --N 24 --m 5
```

**Error handling:** Independently validates both `-N` and `-m`. If either is
not a valid integer or is less than 2, the command prints the corresponding
error message (`N must be an integer >= 2` or `m must be an integer >= 2`) to
stderr and exits with code 1.

**Example output:**

```
Running parameter grid search...
...done (saved results).
```

---

#### `download-sample-data`

Download sample data for the pfun-cma-model package.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| *(none)* | | | No options available. |

**How it works:** Prints a status message indicating that sample data is
being downloaded. Currently a stub implementation that simulates downloading
to `sample_data.csv`.

**Usage:**

```bash
pnpm --filter cli start download-sample-data
```

**Example output:**

```
Downloading sample data for the pfun-cma-model package...
...sample data downloaded to: sample_data.csv
```

---

#### `benchmark`

Run performance benchmarks on the CMA sleep-wake model.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| *(none)* | | | No options available. |

**How it works:** Instantiates a default `CMASleepWakeModel` (no arguments,
uses the internal default for `N`) and calls `.solve()`, then prints a
confirmation message.

**Usage:**

```bash
pnpm --filter cli start benchmark
```

**Example output:**

```
Running benchmarks...
Results saved to benchmark output
```

---

### Programmatic Usage

The CLI module can be imported without triggering automatic argument parsing.
This is useful for tests and for embedding the CLI in larger scripts:

```typescript
// Import the module — no side effects
const cli = await import('cli');
// Parse manually when needed:
// cli.program.parse(['node', 'entry', '--help']);
```

This is achieved via the import guard at the bottom of `src/index.ts`:

```typescript
const isDirectRun = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isDirectRun) {
  program.parse(process.argv);
}
```

The guard compares the current module's file path (resolved via
`import.meta.url`) against `process.argv[1]` — the entry point of the running
process. When the module is loaded as a library via `import()`, the paths
differ and `program.parse()` is never called.

---

### Architecture Notes

- **Dependency chain:** `cli` → `@pfun/api` → `@pfun/core`. The `launch`
  command uses a dynamic `import()` to load the API at runtime, while
  `fit-model`, `generate-scenario`, `run-param-grid`, and `benchmark` use
  direct imports from `@pfun/core`.
- **Path resolution:** Because the package is ESM, it uses
  `fileURLToPath(import.meta.url)` combined with `dirname` and `join` to
  resolve sibling package paths at runtime (e.g., finding `../../api/dist/`
  relative to the CLI dist directory).
- **Commander pattern:** Commands are defined with a fluent
  `.command().description().option().action()` chain. All commands use
  `console.log` / `console.error` for I/O and call `process.exit(1)` on
  fatal errors.
- **Build output:** TypeScript is compiled to ESM (`dist/index.js`) with
  `moduleResolution: "bundler"`, compatible with Node.js ESM imports.
- **Zod dependency:** Although `zod` is listed as a dependency, it is
  available for future validation enhancements; the current commands use
  manual `parseInt` with guard clauses.

### Testing

CLI tests are written with **Vitest** and live in
`packages/cli/src/__tests__/cli.test.ts`.

```bash
pnpm --filter cli test
```

The test suite verifies:

- The module can be imported as an ES module without errors (the import guard
  prevents accidental command parsing during import).

Currently one test is defined:

```typescript
import { describe, it, expect } from 'vitest';

describe('CLI Package', () => {
  it('should export the CLI entry point as an ES module', async () => {
    const cli = await import('../index.js');
    expect(cli).toBeDefined();
  });
});
```

---

## Development

### Scripts

| Command | Description |
|---|---|
| `pnpm install` | Install all workspace dependencies |
| `pnpm build` | Build all packages (recursive) |
| `pnpm test` | Run all tests (vitest run) |
| `pnpm test:watch` | Run tests in watch mode |
| `pnpm --filter @pfun/core build` | Build only the core package |
| `pnpm --filter @pfun/core test` | Run core package tests |
| `pnpm --filter @pfun/api test` | Run API package tests |
| `pnpm --filter @pfun/api dev` | Start API in dev mode with hot reload |
| `pnpm vitest run` | Run all tests from root |

### Workflow

1. **Make changes** in the appropriate package's `src/` directory.
2. **Run relevant tests** — `pnpm --filter @pfun/core test` for core changes.
3. **Build the package** — `pnpm --filter @pfun/core build` (rebuilds `dist/`).
4. **Run all tests** — `pnpm test` to verify nothing is broken.

> **Important:** The `dist/` directory for `@pfun/core` is checked in and must be kept in sync with source. After any source change to core, run `pnpm --filter @pfun/core build` to update `dist/`.

### Adding dependencies

```bash
# Add to a specific package
pnpm --filter @pfun/core add some-package

# Add as dev dependency
pnpm --filter @pfun/core add -D some-package

# Add to root (shared dev dependency)
pnpm add -w -D some-package
```

### Rebuild after source changes

```bash
# Build just core (fastest for iteration)
pnpm --filter @pfun/core build

# Rebuild everything
pnpm build
```

---

## Testing

The project uses **Vitest** (~3.1+) with 23+ tests across 8 test files.

### Test locations

| Package | Test files | What's tested |
|---|---|---|
| `@pfun/core` | `src/__tests__/cma.test.ts` | Implementation B: CMASleepWakeModel.solve(), CMAModelParamsSchema, PFunCMAParamsGrid, generateScenario, engine functions |
| `@pfun/core` | `tests/model.test.ts` | Implementation A: CMASleepWakeModel.run(), runAtTime(), streams, updateParams |
| `@pfun/core` | `tests/calc.test.ts` | Math functions: exp, expitPfun, K, Light, linspace, computeG, etc. |
| `@pfun/core` | `tests/bounds.test.ts` | Bounds constraint class |
| `@pfun/core` | `tests/fit.test.ts` | Model fitting via coordinate descent |
| `@pfun/core` | `tests/params.test.ts` | Parameter metadata and validation |
| `@pfun/api` | `tests/health.test.ts` | Health routes, OpenAPI docs, security middleware |
| `@pfun/api` | `tests/model-params.test.ts` | Model and param route integration |
| `@pfun/api` | `tests/routes.test.ts` | Additional route coverage |
| `cli` | `src/__tests__/cli.test.ts` | CLI module import |

### Running tests

```bash
# All tests
pnpm test

# Watch mode
pnpm test:watch

# Single package
pnpm --filter @pfun/core test

# Core tests with verbose output (CI-friendly)
pnpm --filter @pfun/core test:ci

# API tests
pnpm --filter @pfun/api test
```

Tests import from source (not `dist/`) via Vitest's module resolution, so there's no need to rebuild before running tests during development.

---

## Architecture Notes

### Dual Implementation Design

The core package contains two parallel implementations of the CMA model:

| Aspect | Implementation B (primary) | Implementation A (secondary) |
|---|---|---|
| **Source** | `cma.ts` + `engine.ts` | `model.ts` + `calc.ts` |
| **Params schema** | `cma_model_params.ts` (N=24 default) | `params.ts` (N=1024 default) |
| **Data structures** | `Float64Array`, `Int32Array` | Plain `number[]` arrays |
| **Execution** | `solve()` with pre-allocated typed arrays | `run()` with dynamic array operations |
| **Model class** | `CMASleepWakeModel` (cma.ts) | `CMASleepWakeModel` (model.ts) |
| **Primary consumer** | External library consumers, CLI | `fit.ts` internally, higher-level API |

Both export the same class name (`CMASleepWakeModel`) from the package's `index.ts`, so consumers can choose which to use based on their performance vs. ergonomics needs.

### Key architectural decisions

- **`index.ts`** re-exports from both implementations, with Implementation B listed first (primary/default).
- **`fit.ts`** uses Implementation A's `CMASleepWakeModel` and `params.ts` schema internally.
- **`grid.ts`** and **`llm.ts`** use Implementation B's model and schema.
- The API's Socket.IO handler uses Implementation B's model (`cma.ts`) for streaming.
- The `dist/` files are committed to the repository and must be rebuilt after source changes.
- The CLI dynamically imports the API package at runtime for the `launch` command.

### Parameter schemas

There are two different Zod parameter schemas:

- **`cma_model_params.ts`** (`CMAModelParamsSchema`) — Used by Implementation B. Default `N=24`. `taug` is a single number. `tM` is a fixed 3-element tuple.
- **`params.ts`** (`CMAModelParamsSchema`) — Used by Implementation A. Default `N=1024`. `taug` accepts `number | number[]`. `tM` is a variable-length array. Includes bounded parameter metadata.

---

## License

ISC License. See the root `package.json` for details.
