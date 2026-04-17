---
icon: lucide/image
---

# Demo Gallery

A visual tour of PFun CMA Model's interactive demos, desktop application, and output visualizations.

---

## LLM Scenario Generation

Generate physiologically valid scenarios from natural language. The LLM translates qualitative descriptions into CMA model parameters with clinically relevant recommendations.

![LLM Generate Scenario Demo — input prompt](../assets/img/Screenshot 2026-02-13 at 14-25-06 LLM Generate Scenario Demo ~ PFun Digital Health.png)

![LLM Scenario — generated recommendations](../assets/img/Screenshot 2026-02-13 at 14-28-01 LLM Generate Scenario Demo ~ PFun Digital Health.png)

**→ [LLM Scenarios deep dive](llm-scenarios.md)**

---

## WebSocket Real-Time Streaming

Control model parameters interactively with sliders and see live glucose curve updates via WebSocket:

![WebSocket demo — parameter sliders with live chart](../assets/img/Screenshot 2025-08-10 at 02-36-51 Run-at-Time WebSocket Example.png)

![WebSocket demo — interactive exploration](../assets/img/Screenshot 2025-08-11 at 18-53-40 Run-at-Time WebSocket Example.png)

**→ [WebSocket Streaming deep dive](websocket-streaming.md)**

---

## Parameter Grid Visualization

Precomputed parameter grids stored in DuckDB, visualized to explore the full CMA parameter space:

![CMA parameter grid — multi-dimensional exploration](../assets/img/Screenshot_20260416_201527 cma_grid cma_pgrid.png)

---

## Model Output Visualizations

### CMA Decomposition

Full cortisol-melatonin-adiponectin decomposition from glucose time series:

![CMA decomposition](../assets/img/generated.png)

### Fit Results

Side-by-side model fit (blue) vs. observed data (red):

![24-hour fit result](../assets/img/24hr_fit_result_blueModel_redData.png)

![Dinner fit result](../assets/img/dinner_fit_result_blueModel_redData.png)

### Parameter Tables

Automatically generated parameter tables with qualitative descriptors:

![Parameter table with qualitative descriptions](../assets/img/24hr_fit_result_pretty_markdown_table_with_qualitative_descriptions.png)

---

## Video Demos

### Real-Time Data Streaming

A screencast demonstrating the real-time WebSocket data streaming interface:

<video controls width="100%">
  <source src="../assets/video/Screencast From 2026-01-12 21-58-05 (trimmed).mp4" type="video/mp4">
  <source src="../assets/video/Screencast From 2026-01-12 21-58-05 (trimmed).webm" type="video/webm">
  Your browser does not support the video tag.
</video>

---

## Available Demo Endpoints

When running the dev server (`uv run fastapi dev pfun_cma_model/app.py --port 8001`):

| Demo | URL | Description |
|------|-----|-------------|
| LLM Scenario | `/demo/llm` | Natural language → scenario generation |
| Run-at-Time | `/demo/run-at-time` | WebSocket + Chart.js live plotting |
| Canvas Wave | `/demo/canvas-wave` | HTML5 Canvas wave equation visualization |
| Full Model Run | `/demo/full-model-run` | Complete CMA model with all signals |
| WebGL Plot | `/demo/webgl-demo` | GPU-accelerated real-time plotting |
| Data Stream | `/demo/data-stream` | Server-sent data streaming |
