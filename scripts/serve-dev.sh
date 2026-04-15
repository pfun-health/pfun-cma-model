#!/usr/bin/env bash

# scripts/serve_dev.sh : serve the current version of pfun-cma-model locally with hot-reload

DEV_CERTS_METHOD="${DEV_CERTS_MEHOD:-tailscale}"
SCRIPTS_DIR=$(dirname "$0")

# Load common functions
source "${SCRIPTS_DIR}/_funcs.def.sh"

# Start the development server with full uv sync and serve pfun-cma-model
full_uv_sync && \
    uv run python "${SCRIPTS_DIR}/generate-dev-certs.py" "--${DEV_CERTS_METHOD}"
	serve_pfun_cma_model_dev "$@"
