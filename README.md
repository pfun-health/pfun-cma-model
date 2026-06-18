# pfun-cma-model

## Links (Demos, Homepage)

- [**PFun Homepage**](https://pfun.one/)
- [**[_Live Demo_] PFun Health Tips **](https://cloud.tail38611b.ts.net/demo/llm)

## Overview

### API Description

The `pfun-cma-model` API provides a comprehensive framework for analyzing and modeling the interplay between circadian rhythm, glucose metabolism, and hormonal dynamics. It enables researchers and practitioners to understand how physiological processes influence glucose levels over time.

#### In simple terms, what exactly does it do?!?

A few pithy one-liners:

- **Phase-based dimensionality reduction:** "Included is a well-validated (on ~30million rows of CGM data) phase portrait analysis technique that can compress weeks', months', or even many-years'-worth of glucose time-series data into a minimum-length phase vector (`>= 1024b in memory`)."
- **Interpretable, Quantifiable:** _It provides a way to quickly translate between qualitative ("mood", e.g.) & biophysical neuroendocrine dynamics ("cortisol levels", e.g.)._
- _It provides a high-speed interface for understanding how the circadian rhythm maps to glucose values._

#### Background

- **About the project:** <a href="https://cloud.tail38611b.ts.net/pitch">PFun Glucose ~ App pitch summary</a>
- **Preliminary research summary (includes citations):** <a href="./docs/rendered_pdf/PFun Glucose - Chronometabolic Analysis.pdf">PFun Glucose - Chronometabolic Analysis.pdf</a>

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

## Development notes

- Using `uv` for super fast dependency management, intuitive CLI, and ezpz publishing to pypi.

### Usage notes

#### `nix`, `devenv`

##

	# https://devenv.sh/guides/using-with-flakes/#entering-the-shell
	nix develop --no-pure-eval


#### (dashlane) Inject secrets to create `.env`

```bash

# NOTE: only works if you have dcli (the dashlane CLI) installed locally
$ ./scripts/inject-secrets-env.sh

```

### Convert `docker-compose.yml` to Helm Chart

##

	# convert docker-compose.yml to a Helm Chart (for kubernetes)
	\# kompose convert -c -o pfun-cma-model-chart
	...
	
	# build a binary helm chart package (ready for deployment)
	\# helm package pfun-cma-model-chart --destination dist/pfun-cma-model-chart
	...
	
	# install the helm chart
	\# helm install pfun-cma-model-chart -f dist/pfun-cma-model-chart-<VERSION>.tgz
	...

### (containerized) `docker-compose` environment

#### Complete rebuild & launch

##

	docker compose up -d \
		--build \
		--renew-anon-volumes \
		--remove-orphans

	# ...or with the convenience script:
	./scripts/full-rebuild.sh

#### (Nix) `devenv shell`

##

    # Enter the devenv shell environment (see flake.nix)
    devenv shell
    ...

### (local) `uv` Python Dev environment

#### Debugging the app locally (run as a local FastAPI server)

##

	# Run using fastapi development server
	$ uv run fastapi dev pfun_cma_model/app.py --port 8001

	# Alternatively, use the convenience script
	$ scripts/serve-dev.sh


## Interact with the app via CLI

##

	$ pfun-cma-model generate-scenario --query 'a healthy individual with a tendency to sleep in.'
	{
		"qualitative_description": "This individual is a healthy young adult who is a natural 'night owl'. They have a delayed sleep phase, meaning they tend to go to bed late, around 2:00 AM, and wake up late in the morning, typically after 10:00 AM. Their meal schedule is shifted accordingly, with 'breakfast' often being eaten closer to noon. They are otherwise healthy, with a stable diet and regular activity levels, but their entire daily rhythm, including their natural cortisol cycle, is pushed back by several hours compared to someone with a more conventional sleep schedule.",
    "parameters": {
        "toff": 2.5,
        "d": 0,
        "taup": 1,
        "taug": 1,
        "B": 0.05,
        "Cm": 0
		}
	}

	# fit the CMA model using partial least squares, plot the results
	$ uv run pfun-cma-model run-fit-model --plot

