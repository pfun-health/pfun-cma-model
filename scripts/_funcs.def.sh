#!/usr/bin/env sh

set -e

# _funcs.def.sh : Define common functions used across scripts.

full_uv_sync() {
	# Perform a full uv sync including all extras and specific groups.
	uv sync \
		--reinstall \
		--all-extras \
		--group ollama \
		--group datasette
}

full_uv_sync_gradio() {
	# Perform a full uv sync including all extras and specific groups.
	uv sync \
		--reinstall \
		--all-extras \
		--project=pfun_gradio
}

partial_uv_sync() {
	# Perform a partial uv sync for only the core packages.
	uv sync
}

serve_pfun_cma_model_dev() {
	# [DEV] Serve the pfun_cma_model FastAPI app using the custom 'pfun-cma-model launch' CLI command.
	/usr/bin/env sh -c 'uv run pfun-cma-model launch'
}

serve_pfun_cma_model() {
	# [PROD] Serve the pfun_cma_model FastAPI app using uvicorn with uv.
	/usr/bin/env sh -c 'uv run uvicorn pfun_cma_model.main:app --port 8001 --workers 1 '"--ssl-certfile $(ls certs/*.crt) --ssl-keyfile $(ls certs/*.key)"
}

serve_pfun_gradio() {
	/usr/bin/env sh -c 'cd pfun_gradio; uv run uvicorn pfun_gradio.main:app --port 7860 --workers 2'
}
