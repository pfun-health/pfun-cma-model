#!/usr/bin/env sh

# quick-relaunch-dev.sh

set -e

echo "relaunching dev..."

echo "killing uvicorn..."
pkill -9 uvicorn
echo "...killed using pkill (sleeping for 2s)."
sleep 2s

echo "serving dev..."
./scripts/serve-dev.sh
