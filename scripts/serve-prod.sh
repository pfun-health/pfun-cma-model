#!/usr/bin/env bash

set -e

# scripts/serve-prod.sh : serve the current version of pfun-cma-model (entrypoint for container)

# Load common functions
export REPO_ROOT="${PWD}"
source "${REPO_ROOT}/scripts/_funcs.def.sh"

echo -e "# Serving pfun-cma-model (prod)\n\n"

tstamp="$(date --iso-8601='hours')"
echo -e " + RepoRoot:\t'${REPO_ROOT}'\n + Timestamp:\n${tstamp}\n"
sleep 1s

# Start server full uv sync and serve pfun-cma-model
full_uv_sync
# (the additional args are important, especially for docker entrypoint/command)
serve_pfun_cma_model "$@"
