# pfun-cma-model

## Links (Demos, Homepage)

- [**PFun Homepage**](https://pfun.one/)
- [**Terminal Demo Video**](./DEMO.md) — performance benchmarks + 3D waveform animation

[![CMA Model 3D Waveform Visualization](docs/assets/img/demo-terminal.svg)](./DEMO.md)

## Overview

### API Description

The `pfun-cma-model` API provides a comprehensive framework for analyzing and modeling the interplay between circadian rhythm, glucose metabolism, and hormonal dynamics. It enables researchers and practitioners to understand how physiological processes influence glucose levels over time.

#### In simple terms, what exactly does it do?!?

A few pithy one-liners:

- **Phase-based dimensionality reduction:** "Included is a well-validated (on ~30million rows of CGM data) phase portrait analysis technique that can compress weeks', months', or even many-years'-worth of glucose time-series data into a minimum-length phase vector (`>= 1024b in memory`)."
- **Interpretable, Quantifiable:** _It provides a way to quickly translate between qualitative ("mood", e.g.) & biophysical neuroendocrine dynamics ("cortisol levels", e.g.)._
- _It provides a high-speed interface for understanding how the circadian rhythm maps to glucose values._

#### Background

- **About the project:** <a href="https://pfun-health.github.io/pfun-cma-model">PFun CMA Model Documentation</a>
- **Preliminary research summary (includes citations):** <a href="./docs/pfun-glucose-chronometabolic-analysis.md">Chronometabolic Analysis (Markdown)</a> · <a href="./docs/rendered_pdf/PFun%20Glucose%20-%20Chronometabolic%20Analysis.pdf">PDF</a>

### About this repository

**Generated Cortisol-Melatonin-Adiponectin decomposition (from Glucose time series)**

![Generated Cortisol-Melatonin-Adiponectin decomposition (from Glucose time series).](./results/generated.png)

<div style="border-width: 1px; border-color: #444;">The CMA model leverages physiological modeling principles to decompose glucose time series data into underlying hormonal influences, specifically cortisol, melatonin, and adiponectin. See example notebooks in the live Demo (or in ./examples/notebooks)</div>

### Project Goals

**For detailed development information, check the `TODO.md`:**

- [**TODO.md**](./TODO.md "TODO.md")

## CMA Model Description

#### Model Parameters

| Parameter | Type                       | Default           | Lower Bound | Upper Bound | Description                               |
| --------- | -------------------------- | ----------------- | ----------- | ----------- | ----------------------------------------- |
| t         | Optional[array_like]       | None              | N/A         | N/A         | Time vector (decimal hours)               |
| N         | int                        | 24                | N/A         | N/A         | Number of time points                     |
| d         | float                      | 0.0               | -12.0       | 14.0        | Time zone offset (hours)                  |
| taup      | float                      | 1.0               | 0.5         | 3.0         | Circadian-relative photoperiod length     |
| taug      | float                      | 1.0               | 0.1         | 3.0         | Glucose response time constant            |
| B         | float                      | 0.05              | 0.0         | 1.0         | Glucose Bias constant                     |
| Cm        | float                      | 0.0               | 0.0         | 2.0         | Cortisol temporal sensitivity coefficient |
| toff      | float                      | 0.0               | -3.0        | 3.0         | Solar noon offset (latitude)              |
| tM        | Tuple[float, float, float] | (7.0, 11.0, 17.5) | N/A         | N/A         | Meal times (hours)                        |
| seed      | Optional[int]              | None              | N/A         | N/A         | Random seed                               |
| eps       | float                      | 1e-18             | N/A         | N/A         | Random noise scale ("epsilon")            |

#### Example Fitted Parameters

| Parameter | Value         | Example Description (Human provided)                                           |
| --------- | ------------- | ------------------------------------------------------------------------------ |
| d         | -2.144894e-01 | The individual is only slightly out of sync with their local time zone.        |
| taup      | 4.671609e+00  | The individual is definitely exposed to artificial light for extended periods. |
| taug      | 1.097094e+00  | The individual's glucose response is within a normal range.                    |
| B         | 1.288179e-01  | The individual has a bias towards higher glucose levels.                       |
| Cm        | 0.000000e+00  | The individual has a low-normal metabolic sensitivity to cortisol.             |
| toff      | 0.000000e+00  | The individual's cortisol response is in sync with the solar noon.             |

## Development

### Prerequisites
- Node.js 22+
- pnpm 11+

### Setup

```bash
# Install dependencies
pnpm install

# Build all packages
pnpm build
```

### Run Tests

```bash
pnpm test
```

### Development

```bash
# Watch mode for tests
pnpm test:watch
```

### Package Structure

- `packages/core` - Core CMA model logic
- `packages/api` - API server
- `packages/cli` - CLI tool

