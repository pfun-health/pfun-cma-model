# Documentation Enhancement Summary

## What was created

A comprehensive **16-page** Zensical documentation site for `pfun-cma-model`, replacing the previous 3-page skeleton.

## Documentation Structure

```
docs/
├── index.md                     ← Landing page (logo, all screenshots, architecture)
├── getting-started/
│   ├── installation.md          ← uv/pip/nix install, dep groups, env config
│   └── quickstart.md            ← 7 end-to-end workflows
├── model/
│   ├── overview.md              ← CMA math, mermaid diagrams, signal theory
│   ├── parameters.md            ← All 11 params, bounds, qualitative descriptors
│   └── fitting.md               ← fit_model pipeline, CMAFitResult schema
├── demos/
│   ├── gallery.md               ← Visual tour: all screenshots + video embeds
│   ├── llm-scenarios.md         ← Sequence diagram, backends, prompt engineering
│   ├── websocket-streaming.md   ← Architecture, video demo, Socket.IO code
│   ├── qt-gui.md                ← Class diagram, mixins, healthcheck/auto-retry
│   └── notebooks.md             ← 6 notebooks + 5 code samples
├── cli.md                       ← All 6 CLI commands with option tables
├── api.md                       ← Route groups, Swagger/ReDoc, client generation
├── deployment.md                ← Docker, K8s/Helm, Cloud Run, domains
├── security.md                  ← 59 tests, middleware config, CSP headers
└── contributing.md              ← Dev setup, ruff/mypy, test guidelines
```

## Media Assets Used

**14 images** and **3 videos** from the existing `examples/` and `results/` directories were copied to `docs/assets/` and embedded across pages:

| Asset | Used In |
|-------|---------|
| 5 screenshots (LLM demo, WebSocket, grid) | index, gallery, demos |
| CMA decomposition (`generated.png`) | index, gallery, model overview |
| 3 fit result plots | index, fitting, gallery |
| Parameter table screenshot | index, parameters, gallery |
| PFun logo (`pfunCelticFlower.png`) | index |
| Screencast video (mp4 + webm) | gallery, websocket-streaming |

## Mermaid Diagrams Created

- **CMA signal flow** (model/overview.md) — Circadian inputs → Hormonal signals → Metabolic output
- **Fitting pipeline** (model/fitting.md) — Raw data → format → estimate → curve_fit → result
- **LLM scenario sequence** (demos/llm-scenarios.md) — User → API → LLM → CMA → Response
- **WebSocket architecture** (demos/websocket-streaming.md) — Browser ↔ Socket.IO ↔ CMA
- **Qt GUI class diagram** (demos/qt-gui.md) — Mixins, overlay, theme relationships

## Configuration Changes

Updated [zensical.toml](file:///home/robbiec/Git/pfun-cma-model/zensical.toml) with:
- Rich hierarchical navigation (5 top-level sections)
- Navigation tabs, expand, TOC follow
- Search suggestions and highlighting

## Build & Preview

```bash
uv run zensical serve   # Live preview
uv run zensical build   # Static site → site/
```
