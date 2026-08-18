#!/usr/bin/env bash

set -e

# scripts/full-rebuild.sh : Script to perform a full rebuild of the uv environment and relaunch docker compose services.

# Load common functions
source "$(dirname "$0")/_funcs.def.sh"

docker compose down

export CURRENT_VERSION NEW_VERSION

echo -e "cleaning old dists..."
rm -vrf "$(dirname "$0")/dist"
sleep 1s

echo -e "relaunching compose services..."
sleep 1s
docker compose up -d \
	--remove-orphans \
	--renew-anon-volumes \
	--build
sleep 1s

echo -e "...done relaunching compose services."
