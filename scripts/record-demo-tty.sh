#!/usr/bin/env bash

set -e

# record-demo-tty.sh

rm --force results/demo.cast results/demo.gif

asciinema rec --window-size '80x25' \
    -c "uv run python scripts/demo-video.py" \
    results/demo.cast

sleep 0.5s

agg results/demo.cast results/demo.gif

echo "see: results/demo.cast results/demo.gif"
