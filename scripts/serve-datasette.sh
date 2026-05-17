#!/usr/bin/env bash

set -e

# scripts/serve-datasette.sh

echo -e "running datasette in the background (using metadata.json config)"
sleep 1s

pkill -9 datasette

rm logs/datasette.log; touch logs/datasette.log
sleep 1s

echo -e "starting datasette..."
sleep 1s

nohup \
    uv run datasette \
    --metadata ./metadata.json \
    logs/datasette.log

sleep 1s;
cat logs/datasette.log
