#!/usr/bin/env bash

set -e

# scripts/run-tests.sh:
# Run CLI tests and limit output to first 80 lines

uv run \
    pytest -v --tb=short 2>&1 | head -80