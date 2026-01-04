#!/usr/bin/env bash

# scripts/serve_dev.sh : serve the current version of pfun-cma-model locally with hot-reload

# Load common functions
source "$(dirname "$0")/_funcs.def.sh"

serve_pfun_cma_model() {
	uv run pfun-cma-model launch
}

serve_pfun_gradio() {
	/usr/bin/env bash -c 'cd packages/pfun_gradio && uv run uvicorn pfun_gradio.main:app'
}


# Start the development server with full uv sync and serve pfun-cma-model
full_uv_sync && \
	serve_pfun_cma_model