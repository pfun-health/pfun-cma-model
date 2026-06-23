#!/usr/bin/env sh

set -e

# scripts/run-benchmarks.sh :
# run benchmarks using pyperformance

OUTPUT_JSON_PATH="results/benchmarks/py312-uv.json"

echo "Running benchmarks (pyperformance)..."
echo "Output saved to: '${OUTPUT_JSON_PATH}'"
echo "######################################"

uv run \
   pyperformance run \
   -o "${OUTPUT_JSON_PATH}"
