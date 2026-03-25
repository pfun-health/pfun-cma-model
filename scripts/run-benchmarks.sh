#!/usr/bin/env sh

# scripts/run-benchmarks.sh :
# run benchmarks using pyperformance

uv run \
   pyperformance run \
   --python="$(uv python dir)/bin/python" \
   -o "py312-uv.json"
