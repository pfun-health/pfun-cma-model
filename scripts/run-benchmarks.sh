#!/usr/bin/env sh

# scripts/run-benchmarks.sh :
# run benchmarks using pyperformance

uv run pyperformance run --python=$(which python3.12) -o py312-uv.json
