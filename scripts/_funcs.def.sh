#!/usr/bin/env bash

set -e

# _funcs.def.sh : Define common functions used across scripts.

full_uv_sync() {
	# Perform a full uv sync including all extras and specific groups.
	uv sync --all-extras \
		--group perplexity \
		--group gradio
}
