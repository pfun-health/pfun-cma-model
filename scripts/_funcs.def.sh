#!/usr/bin/env sh

set -e

# _funcs.def.sh : Define common functions used across scripts.

full_uv_sync() {
    # Perform a full uv sync including all extras and specific groups.
    uv sync \
       --reinstall \
       --all-extras \
       --group ollama \
       --group gradio
}

partial_uv_sync() {
	# Perform a partial uv sync for only the core packages.
	uv sync
}
