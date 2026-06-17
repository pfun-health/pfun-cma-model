#!/usr/bin/env sh

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

full_uv_sync_qt_gui() {
	# Perform a full uv sync for the qt-gui package.
	local GUI_DIR=$(realpath "$SCRIPT_DIR/../packages/pfun_qt_gui")
	cd "$GUI_DIR" && uv sync \
		--reinstall \
		--all-extras \
		--link-mode copy
}

full_uv_sync_gradio() {
	# Perform a full uv sync including all extras and specific groups.
	uv sync \
		--reinstall \
		--all-extras \
		--project=pfun_gradio \
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

serve_pfun_gradio() {
	/usr/bin/env sh -c "cd pfun_gradio; uv run uvicorn pfun_gradio.main:app --port 7860 --workers 1 $(get_ssl_args) $*"
}
