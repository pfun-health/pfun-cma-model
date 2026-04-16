#!/usr/bin/env sh

# quick-relaunch-prod.sh

set -e

echo "relaunching prod..."

echo "killing uvicorn..."
pkill -9 uvicorn
echo "...killed using pkill (sleeping for 2s)."
sleep 2s

echo "serving prod..."
./scripts/serve-prod.sh full
