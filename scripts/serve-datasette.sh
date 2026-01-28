#!/usr/bin/env bash

set -e

# scripts/serve-datasette.sh

echo -e "running datasette in the background (using metadata.json config)"
nohup uv run datasette --metadata ./metadata.json &
