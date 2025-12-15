# pfun-cma-model

## Overview

### Description

The `pfun-cma-model` API provides a comprehensive framework for analyzing and modeling the interplay between circadian rhythm, glucose metabolism, and hormonal dynamics. It enables researchers and practitioners to understand how physiological processes influence glucose levels over time.

#### What does the `pfun-cma-model` API do???????

_"It makes it easy to understand how glucose works... in clear english, and in medical-ese."_

_"It provides a high-speed interface for understanding how the circadian rhythm maps to glucose values."_

#### Clinical literature search

**<a href="./docs/rendered_pdf/PFun Glucose - Chronometabolic Analysis.pdf">PFun Glucose - Chronometabolic Analysis.pdf</a>**

### About this repository

**Generated Cortisol-Melatonin-Adiponectin decomposition (from Glucose time series)**

![Generated Cortisol-Melatonin-Adiponectin decomposition (from Glucose time series).](./results/generated.png)

<div style="border-width: 1px; border-color: #444;">The CMA model leverages physiological modeling principles to decompose glucose time series data into underlying hormonal influences, specifically cortisol, melatonin, and adiponectin. See example notebooks in the live Demo (or in ./examples/notebooks)</div>

### Next steps

+ compare LLMs via actor-critic framework.
+ measure relative perplexity, NLP performance metrics.
  + KS Test to quantify the hypothesis space;
    + Predicted: neural dynamics of the learning layer (LLM orchestration layer) can be measured & understood as in silico "twin" populations of corticostriatal interneurons.

## Links (Demos, Homepage)

+ [Live Web Demos](https://cloud.tail38611b.ts.net/)
+ [PFun Homepage](https://pfun.one/)

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

#### Example ChatGPT Output

```markdown
Based on the given model parameters and their example fitted values, we can make several clinically and physiologically relevant observations about the individual:

1. **Time Zone Offset (d)**: The value is -0.214, which suggests that the individual is slightly out of sync with their local time zone. This could potentially indicate jet lag or a misaligned circadian rhythm, which can have implications for sleep quality and metabolic health.

2. **Circadian-relative Photoperiod Length (taup)**: The value is 4.67, which is significantly higher than the default of 1.0 and also exceeds the upper bound. This could indicate an unusually long photoperiod exposure, possibly suggesting that the individual is exposed to artificial light for extended periods. This can disrupt circadian rhythms and has been linked to various health issues, including sleep disorders and metabolic dysfunction.

3. **Glucose Response Time Constant (taug)**: The value is 1.097, which is close to the default. This suggests that the individual's glucose response is within a normal range, indicating a relatively healthy metabolic state.

4. **Glucose Bias Constant (B)**: The value is 0.129, which is higher than the default of 0.05. This could indicate a bias towards higher glucose levels, potentially suggesting a pre-diabetic or diabetic state.

5. **Cortisol Temporal Sensitivity Coefficient (Cm)**: The value is -1.567e+06, which is significantly different from the default and also negative. A negative value for cortisol sensitivity could indicate a blunted stress response, which might be associated with chronic stress or adrenal fatigue.

6. **Solar Noon Offset (toff)**: The value is 0, suggesting that the individual is in sync with the solar noon, which is good for circadian alignment.

7. **Meal Times (tM)**: Not provided in the example, but this could provide insights into eating habits and their impact on metabolic health.

8. **Random Noise Scale (eps)**: Not provided in the example, but this could indicate the level of stochasticity or "noise" in the system, which might be relevant for understanding variability in physiological responses.

Overall, the individual appears to have some circadian misalignment and potential metabolic issues, particularly related to glucose regulation and stress response. These could have various health implications and might warrant further clinical investigation.
```

#### Example Non-PFun ChatGPT Response

```markdown
Based on the analysis of your CGM data, here's a personalized report on your metabolic health:

- **Average Glucose:** 120.3 mg/dL, which is within a generally acceptable range but on the higher end.
- **Minimum Glucose:** 95.0 mg/dL, indicating good management of low glucose episodes.
- **Maximum Glucose:** 262.0 mg/dL, suggesting some episodes of significantly elevated glucose.
- **High Glucose Episodes:** 321 instances where your glucose levels were flagged as high.
- **Low Glucose Episodes:** Occurrences of low glucose are less frequent.
- **Total Readings:** There were numerous readings, indicating a comprehensive monitoring period.
- **Standard Deviation of Glucose:** 17.7, which reflects moderate variability in your glucose levels.

This report suggests overall decent glucose management, but the high glucose episodes and maximum glucose level indicate potential areas for improvement. Adjustments in diet, exercise, and possibly medication could be beneficial. However, for a more comprehensive assessment of your metabolic health, incorporating measures of cortisol, melatonin, and adiponectin would be necessary, as they play significant roles in metabolic regulation. Always consult with your healthcare provider for personalized medical advice.
```

#### Example ChatGPT Response with PFun Parameters

```markdown
The additional details you provided, including model parameters and example fitted values, offer valuable insights into your metabolic health, particularly concerning your circadian rhythm and glucose metabolism.

1. **Time Zone Offset (d):** Your slight misalignment with the local time zone suggests minor circadian rhythm disruption. This could be due to travel, work schedule, or lifestyle habits.

2. **Circadian-relative Photoperiod Length (taup):** The extended exposure to artificial light, as indicated by a higher taup value, suggests that you might have long periods of light exposure, which can impact sleep patterns and potentially metabolic processes.

3. **Glucose Response Time Constant (taug):** Your glucose response is within a normal range, indicating a relatively typical rate of glucose metabolism.

4. **Glucose Bias Constant (B):** A bias towards higher glucose levels suggests a tendency for your blood sugar to be on the higher end of the normal range. This could be influenced by diet, physical activity, and overall health.

5. **Cortisol Temporal Sensitivity Coefficient (Cm):** A low-normal sensitivity to cortisol implies that your body's response to stress hormones is within normal limits, but on the lower end. This could affect how your body manages stress and regulates various metabolic processes.

6. **Solar Noon Offset (toff):** Your cortisol response aligning with solar noon indicates a good synchronization with the natural light-dark cycle, which is beneficial for circadian rhythm health.

7. **Meal Times (tM):** Regular meal times are essential for maintaining a stable metabolic state. Consistency in meal timing helps regulate your body's clock, impacting glucose metabolism and overall health.

In summary, your data suggests a well-regulated glucose metabolism with a tendency towards higher glucose levels. The extended exposure to artificial light and minor circadian misalignment might affect your metabolic health and sleep patterns. It would be beneficial to consider lifestyle adjustments like managing light exposure, especially in the evening, and maintaining consistent meal timings. Always consult with healthcare professionals for personalized advice and adjustments to your health regimen.
```

## Development notes

+ Look at websockets
+ Using `uv` for super fast dependency management, usage, and publishing.

### Quick start

#### (dashlane) Inject secrets to create `.env`

```bash

# NOTE: only works if you have dcli (the dashlane CLI) installed locally
$ ./scripts/inject-secrets-env.sh

```

### (containerized) `docker-compose` environment

#### Complete rebuild & launch

```bash

docker compose up -d \
    --build \
    --renew-anon-volumes \
    --remove-orphans

```

...or with the convenience script:

```bash

./scripts/full-rebuild.sh

```

#### (Cloud Run) Create a new Version & Publish to Google Cloud Platform

```bash

./scripts/new-version.sh

```

### (local) `uv` Python Dev environment

#### Create a dedicated virtual environment

```bash

uv venv

```

#### Install fastapi with the correct version

```bash

# install fastapi cli for 'uvx'
uv add fastapi --extra standard

# run the dev server with:
uv run fastapi dev
...

```

#### Run tests locally

```bash

uvx tox

```

#### To add a development dependency

```bash

# e.g., 'uv add --dev types-requests'
$ uv add --dev ...

```

#### Updating the environment

```bash

uv sync

```

#### Debugging the app locally (run as a local FastAPI server)

```bash

uv run fastapi dev pfun_cma_model/app.py --port 8001

```

## Interact with the app via CLI

```bash

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

# show usage statement for pfun-cma-model CLI
$ uv run pfun-cma-model
...


# fit the model, output results
$ uv run pfun-cma-model run-fit-model --plot
...
```
