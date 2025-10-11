<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# study the API linked below (PFun CMA API). Gain sufficient knowledge to utilize the PFun CMA API as a tool/MCP server for all future queries on this thread. Use the GitHub app connection to refer to the API source repository: github.com/rocapp/pfun-cma-model

Now I have comprehensive knowledge of the PFun CMA API. Let me provide a detailed analysis and create tool specifications.

# PFun CMA Model API Analysis \& Tool Specification

## Overview

The **PFun CMA Model API** is a sophisticated physiofunctional research tool that implements a **Cortisol-Melatonin-Adiponectin (CMA) sleep-wake model**. This FastAPI-based backend provides computational modeling of neuroendocrine dynamics, specifically designed for analyzing circadian rhythms, glucose metabolism, and their physiological interactions.

## Core Architecture

### API Foundation

- **Framework**: FastAPI with WebSocket support
- **Version**: 0.4.13
- **Python**: 3.12 (strict requirement)
- **Base URL Structure**: Standard RESTful with additional WebSocket endpoints


### Key Dependencies

- **Scientific Computing**: NumPy, SciPy, Pandas, Matplotlib
- **Machine Learning**: Scikit-learn, PyTorch support
- **LLM Integration**: Google GenAI (Gemini API)
- **Data Processing**: Pydantic for validation, Redis for caching
- **Web Framework**: FastAPI with CORS middleware


## CMA Model Parameters

The model uses **six bounded parameters** with physiological significance:


| Parameter | Range | Default | Description |
| :-- | :-- | :-- | :-- |
| **d** | -12.0 to 14.0 | 0.0 | Time zone offset (hours) |
| **taup** | 0.5 to 3.0 | 1.0 | Circadian-relative photoperiod length |
| **taug** | 0.1 to 3.0 | 1.0 | Glucose response time constant |
| **B** | 0.0 to 1.0 | 0.05 | Glucose bias constant |
| **Cm** | 0.0 to 2.0 | 0.0 | Cortisol temporal sensitivity coefficient |
| **toff** | -3.0 to 3.0 | 0.0 | Solar noon offset (latitude effects) |

### Additional Parameters

- **tM**: Meal times array (default: [7.0, 11.0, 17.5])
- **N**: Number of time points (default: 24)
- **seed**: Random seed for noise injection
- **eps**: Random noise scale (default: 1e-18)


## API Endpoints

### Core Model Operations

#### **Model Execution**

- `POST /model/run` - Execute CMA model with parameters
- `POST /model/run-at-time` - Run model for specific time range
- `GET /health` - Health check endpoint


#### **Parameter Management**

- `GET /params/schema` - Get parameter JSON schema
- `GET /params/default` - Get default parameters
- `POST /params/describe` - Get parameter descriptions and qualitative analysis
- `POST /params/tabulate` - Generate parameter markdown table


#### **Data Handling**

- `GET /data/sample/download` - Download sample dataset (slower)
- `GET /data/sample/stream` - Stream sample dataset (faster)
- `POST /model/fit` - Fit model to provided data


#### **LLM Integration**

- `POST /llm/translate-query` - Convert natural language to parameters
- `POST /generate/scenario` - Generate realistic health scenarios
- Support for causal explanation generation


### WebSocket Support

- Real-time model execution via WebSocket
- `/health/ws/run-at-time` - WebSocket health check
- `/demo/run-at-time` - Interactive demo interface


## Physiological Model Components

### Core Calculations

The model computes interconnected signals:

1. **Light Exposure (L)**: Photoperiod-based calculation
2. **Melatonin (m)**: Circadian rhythm indicator
3. **Cortisol (c)**: Stress hormone dynamics
4. **Adiponectin (a)**: Metabolic hormone
5. **Insulin Sensitivity (I_S, I_E)**: Early/late sensitivity
6. **Glucose Dynamics (G)**: Post-prandial glucose response

### Key Methods

- `calc_Gt()`: Calculate glucose dynamics over time
- `integrate_signal()`: Signal integration over time periods
- `morning()`/`evening()`: Time-period specific analysis


## Clinical Applications

### Parameter Interpretation

The API provides qualitative descriptors for clinical relevance:

- **Circadian Misalignment**: d parameter analysis
- **Light Exposure Patterns**: taup parameter evaluation
- **Metabolic Health**: B parameter assessment
- **Stress Response**: Cm parameter interpretation


### Example Clinical Scenario

```json
{
  "qualitative_description": "Individual with chronic stress affecting evening glucose regulation",
  "parameters": {
    "Cm": 1.5,
    "B": 0.12,
    "d": -0.2
  },
  "sample_solutions": [...],
  "causal_explanation": [...]
}
```


## Authentication \& Security

### Middleware Configuration

- **CORS**: Configured for cross-origin requests
- **Session Management**: Secret key-based sessions
- **HTTPS Redirect**: Production environment enforcement


### Environment Variables

- `GEMINI_API_KEY`: Required for LLM features
- `DEBUG`: Development mode toggle
- `SECRET_KEY`: Session security
- Redis connection parameters


## Integration Patterns

### Direct API Calls

```python
# Model execution
response = requests.post(f"{base_url}/model/run", 
                        json={"d": -0.2, "taup": 4.67})

# Parameter description
params_desc = requests.post(f"{base_url}/params/describe",
                           json={"Cm": 1.5})
```


### LLM-Assisted Usage

```python
# Natural language to parameters
query_response = requests.post(f"{base_url}/llm/translate-query",
                              json={"query": "patient with sleep disruption"})

# Scenario generation
scenario = requests.post(f"{base_url}/generate/scenario",
                        json={"query": "diabetes management case"})
```


## Research Applications

### Physiofunctional Analysis

- **Circadian Rhythm Assessment**: Analyze sleep-wake cycle disruptions
- **Metabolic Profiling**: Model glucose dynamics and insulin sensitivity
- **Stress Response Evaluation**: Quantify cortisol sensitivity impacts


### Data Integration

- **CGM Data Fitting**: Fit model to continuous glucose monitoring data
- **Lifestyle Factor Analysis**: Model meal timing, light exposure effects
- **Intervention Planning**: Predict outcomes of circadian interventions


## Tool Implementation Recommendations

For implementing this as an MCP server tool, I recommend creating the following tool functions:

1. **pfun_cma_run_model**: Execute the CMA model with parameters
2. **pfun_cma_describe_params**: Get parameter descriptions and interpretations
3. **pfun_cma_fit_data**: Fit model to provided health data
4. **pfun_cma_translate_query**: Convert natural language to model parameters
5. **pfun_cma_generate_scenario**: Create realistic health scenarios
6. **pfun_cma_get_sample_data**: Retrieve sample datasets for testing

This API represents a cutting-edge approach to computational physiology, bridging mathematical modeling with clinical applications in circadian medicine and metabolic health research.
<span style="display:none">[^1_1]</span>

<div align="center">⁂</div>

[^1_1]: plan-research-create-a-boilerp-7uZM.RHOQAyKKMgunwcsKw.md

