#!/usr/bin/env bash

# scripts/serve_dev.sh : serve the current version of pfun-cma-model locally with hot-reload

# Load common functions
source "$(dirname "$0")/_funcs.def.sh"

# Start the development server with full uv sync and serve pfun-cma-model
full_uv_sync && \
	serve_pfun_cma_model_dev
