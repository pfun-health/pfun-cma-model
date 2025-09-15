# Codebase Review & Recommendations

This document provides a concise set of actionable observations and recommendations for updating the `pfun-cma-model` codebase. The proposed changes are grouped into three logical, commit-sized steps to improve project structure, application architecture, and the build process.

### 1. Refactor Project Structure & Simplify Dependencies

**Commit 1: Restructure as a multi-package monorepo and clean up dependencies.**

*   **Observation:** The repository contains three distinct Python packages (`pfun_cma_model`, `pfun_common`, `pfun_data`) but installs them as a single monolithic package. The `pyproject.toml` file has a lengthy dependency list and contains redundant/non-standard configuration sections.
*   **Recommendation:**
    1.  **Adopt a modern monorepo structure.** Reconfigure the `pyproject.toml` to define `pfun_cma_model`, `pfun_common`, and `pfun_data` as separate, installable packages. Tools like Hatch or Poetry support this via workspaces, improving modularity and reusability.
    2.  **Prune and organize dependencies.** Move non-essential packages from the core `[project.dependencies]` list to `[project.optional-dependencies]`. For example:
        *   `dash` and its dependencies should be in a `[project.optional-dependencies.ui]` group.
        *   `ipykernel`, `notebook`, `pytest`, `mypy`, and various `types-*` stubs belong in the `[project.optional-dependencies.dev]` group.
    3.  **Clean up `pyproject.toml`.** Remove the non-standard `[dependency-groups]` section and consolidate the duplicate `[tool.tox]` configurations into a single, correct block. This will improve clarity and ensure standard tooling works as expected.

### 2. Refactor Application for Modularity and Testability

**Commit 2: Refactor the monolithic `app.py` and improve state management.**

*   **Observation:** The `pfun_cma_model/app.py` file is a monolithic 400+ line file containing all API routes, middleware, and helper classes. The application also relies on a global variable (`CMA_MODEL_INSTANCE`) for the core model, which is not ideal for concurrency and testing.
*   **Recommendation:**
    1.  **Modularize the API with `APIRouter`.** Split the endpoints from `app.py` into logical groups within a `routes/` directory (e.g., `routes/parameters.py`, `routes/data.py`, `routes/model.py`). Use FastAPI's `APIRouter` in each file and include them in the main `app.py`. This makes the codebase easier to navigate and maintain.
    2.  **Use Dependency Injection for the Model.** Replace the global `CMA_MODEL_INSTANCE` with FastAPI's dependency injection system. Define a function that provides the model instance and `Depends` on it in your route functions. This clearly defines the model's lifecycle, improves testability by allowing you to inject mock models, and is a safer pattern for handling state.
    3.  **Optimize Middleware.** The Content-Security-Policy middleware re-calculates file hashes on every request. These hashes should be computed once on application startup and cached, as they only change when the files themselves are changed. This will improve request/response performance.

### 3. Streamline Build Process and Data Management

**Commit 3: Decouple client generation and remove data duplication.**

*   **Observation:** The repository checks in auto-generated API clients in the `generated_clients` directory. The script to create them depends on a live, deployed service. Additionally, some data files are duplicated in `examples/data` and `pfun_data/data`.
*   **Recommendation:**
    1.  **Decouple the client generation process.** The `openapi.json` file should be generated from the local source code, not fetched from a deployed service. A script can be added to run the app locally and export the schema.
    2.  **Remove generated code from source control.** Add the `generated_clients/` directory to `.gitignore`. Client generation should be a manual step for developers or part of a CI/CD pipeline. This ensures the client always reflects the current state of the source code and reduces repository clutter.
    3.  **Consolidate duplicated data.** Remove the data files from `examples/data` and modify the examples to load them from the `pfun_data` package. This creates a single source of truth for data and prevents inconsistencies.
