---
icon: lucide/sliders-horizontal
---

# Model Parameters

## Bounded Parameters

The CMA model has six primary **bounded** parameters that are optimized during curve fitting. Each parameter has physiological meaning and defined bounds:

| Parameter | Default | Lower | Upper | Step | Description |
|-----------|---------|-------|-------|------|-------------|
| `d` | 0.0 | -12.0 | 14.0 | 0.025 | **Time zone offset** (hours) — Photoperiod offset relative to solar noon |
| `taup` (τ_p) | 1.0 | 0.5 | 3.0 | 0.044 | **Photoperiod duration** — Estimated hours of light vs. darkness |
| `taug` (τ_g) | 1.0 | 0.1 | 3.0 | 0.039 | **Glucose response time constant** — Rate of postprandial glucose metabolism |
| `B` | 0.05 | 0.0 | 1.0 | 0.013 | **Glucose baseline constant** — Correlates with basal glucose / A1C |
| `Cm` | 0.0 | 0.0 | 2.0 | 0.025 | **Cortisol sensitivity coefficient** — Influence of cortisol on glucose variability |
| `toff` | 0.0 | -3.0 | 3.0 | 0.000 | **Solar noon offset** — Chronotype alignment with light-dark cycle |

## Unbounded Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `t` | Auto-generated | Time vector (decimal hours, 0–24) |
| `N` | 1024 | Number of time points |
| `tM` | `[7.0, 11.0, 17.5]` | Meal times (hours) |
| `seed` | `None` | Random seed (enables noise via `eps`) |
| `eps` | `1e-18` | Random noise scale ("epsilon") |

## Parameter Interpretation

Each bounded parameter generates a **qualitative descriptor** based on its standardized error relative to the midpoint:

```python
from pfun_cma_model import CMAModelParams

params = CMAModelParams(Cm=1.5, B=0.001, taug=0.4)

# Generate qualitative descriptor
print(params.generate_qualitative_descriptor("Cm"))
# → "Very High"

print(params.describe("Cm"))
# → "Cortisol sensitivity coefficient... (Very High)"
```

The descriptor mapping:

| Condition | Descriptor |
|-----------|-----------|
| `|serr| < 0.1` | **Normal** |
| `serr ≤ -0.1` | **Low** |
| `serr ≥ 0.1` | **High** |
| `|serr| ≥ 0.23` | **Very** (prefix) |

Where `serr = (value - midpoint) / (upper - lower)`.

## Markdown Table Generation

Generate formatted parameter tables programmatically:

```python
params = CMAModelParams(Cm=1.5, B=0.001, taug=0.4)

# Markdown table
print(params.generate_markdown_table(output_fmt="md"))

# HTML table (for web rendering)
html = params.generate_markdown_table(output_fmt="html")

# JSON (for API responses)
json_str = params.generate_markdown_table(output_fmt="json")
```

![Parameter table with qualitative descriptions](../assets/img/24hr_fit_result_pretty_markdown_table_with_qualitative_descriptions.png)

## Example: Fitted Parameters

Here is an example of fitted parameters for a real individual, with human-provided interpretations:

| Parameter | Fitted Value | Interpretation |
|-----------|-------------|----------------|
| `d` | -0.215 | Slightly out of sync with local time zone |
| `taup` | 4.672 | Extended exposure to artificial light |
| `taug` | 1.097 | Glucose response within normal range |
| `B` | 0.129 | Bias toward higher glucose (elevated A1C) |
| `Cm` | 0.000 | Low-normal cortisol sensitivity |
| `toff` | 0.000 | Cortisol in sync with solar noon |

## Using Parameters in Code

```python
from pfun_cma_model import CMASleepWakeModel, CMAModelParams

# Create params object
params = CMAModelParams(
    d=-0.215,
    taup=4.672,
    taug=1.097,
    B=0.129,
    Cm=0.0,
    toff=0.0,
    tM=[7.0, 12.0, 18.0],
    N=288,
)

# Initialize model with params
cma = CMASleepWakeModel(config=params)
df = cma.run()

# Access bounded parameters
print(cma.bounded_params_as_dict)
# {'d': -0.215, 'taup': 4.672, 'taug': 1.097, 'B': 0.129, 'Cm': 0.0, 'toff': 0.0}

# Update parameters dynamically
cma.update(Cm=1.5, B=0.3)
df_updated = cma.run()
```

## Bounds & Constraint Handling

Parameters are automatically constrained to their bounds during updates:

```python
cma = CMASleepWakeModel()

# This will be clipped to the valid range
cma.update(d=100.0)  # d is clipped to 14.0 (upper bound)
print(cma.d)  # → 14.0

# Custom bounds can be set
cma.update_bounds(keys=["taup"], lb=[0.1], ub=[5.0])
```

**→ Next: [Model Fitting](fitting.md)**
