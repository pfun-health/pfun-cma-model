#!/usr/bin/env bash

set -e

# _funcs.def.sh : Define common functions used across scripts.

full_uv_sync() {
	# Perform a full uv sync including all extras and specific groups.
	uv sync \
		--reinstall \
		--all-extras \
		--group llm \
		--group datasette \
		--link-mode copy
}

partial_uv_sync() {
	# Perform a partial uv sync for only the core packages.
	uv sync
}

get_ssl_args() {
	# Get SSL arguments for uvicorn from the certs directory.
	local certfile=$(ls certs/*.crt 2>/dev/null || true)
	local keyfile=$(ls certs/*.key 2>/dev/null || true)
	if [ -n "$certfile" ] && [ -n "$keyfile" ]; then
		echo "--ssl-certfile $certfile --ssl-keyfile $keyfile"
	else
		echo ""
	fi
}

serve_pfun_cma_model_dev() {
	# [DEV] Serve the pfun_cma_model FastAPI app using the custom 'pfun-cma-model launch' CLI command.
	/usr/bin/env sh -c "uv run pfun-cma-model launch $(get_ssl_args) $*"
}

serve_pfun_cma_model() {
	# [PROD] Serve the pfun_cma_model FastAPI app using uvicorn with uv.
	/usr/bin/env sh -c "uv run uvicorn pfun_cma_model.main:app --port 8001 --workers 1 $(get_ssl_args) $*"
}
