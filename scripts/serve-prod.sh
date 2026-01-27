#!/usr/bin/env bash

# scripts/serve-prod.sh : serve the current version of pfun-cma-model (no need for docker)

# Load common functions
export REPO_ROOT=$(dirname "$0")
source "${REPO_ROOT}/_funcs.def.sh"

# Start server full uv sync and serve pfun-cma-model, pfun-gradio
if [ "$1" = 'cma' ]; then
    full_uv_sync; serve_pfun_cma_model
else
    if [ "$1" = 'gradio' ]; then
	full_uv_sync_gradio; serve_pfun_gradio
    else
	if [ "$1" = 'full' ]; then
	    nohup /bin/bash -c 'pkill -9 uvicorn; sleep 1s; cd '"${REPO_ROOT}"'. ./.venv/bin/activate; scripts/serve-prod.sh cma & scripts/serve-prod.sh gradio' &
	else
	    echo 'no server selected (should be either gradio or cma)'
	fi
    fi
fi
