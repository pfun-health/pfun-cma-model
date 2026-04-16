#!/usr/bin/env bash

# scripts/serve-prod.sh : serve the current version of pfun-cma-model (no need for docker)

# Load common functions
export REPO_ROOT="${PWD}"
source "${REPO_ROOT}/scripts/_funcs.def.sh"

echo -e "RepoRoot:\t'${REPO_ROOT}'\n\nTimestamp:\n$(timedatectl)\n"

# Start server full uv sync and serve pfun-cma-model, pfun-gradio
if [ "$1" = 'cma' ]; then
    full_uv_sync; serve_pfun_cma_model
else
    if [ "$1" = 'gradio' ]; then
		full_uv_sync_gradio; serve_pfun_gradio
    else
	if [ "$1" = 'full' ]; then
	    nohup /usr/bin/env sh -c "echo 'serving...'; . ./.venv/bin/activate; ${REPO_ROOT}/scripts/serve-prod.sh cma" &
	else
	    echo 'no server selected (should be either gradio or cma)'
	fi
    fi
fi
