#!/usr/bin/env bash

# scripts/serve_dev.sh : serve the current version of pfun-cma-model locally with hot-reload

uv sync --active --all-extras --group perplexity --group gradio &&

serve_pfun_cma_model() {
	uv run pfun-cma-model launch
}

serve_pfun_gradio() {
	/usr/bin/env bash -c 'cd packages/pfun_gradio && uv run uvicorn pfun_gradio.main:app'
}



serve_pfun_cma_model