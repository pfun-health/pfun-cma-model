#!/usr/bin/env bash

set -e

# scripts/full-rebuild.sh : Script to perform a full rebuild of the uv environment and relaunch docker compose services.

# Load common functions
source "$(dirname "$0")/_funcs.def.sh"

docker compose down

export CURRENT_VERSION NEW_VERSION

"$(dirname "$0")/rebuild-uv.sh" "$@"

echo -e "relaunching compose services..."
sleep 1s
docker compose up -d \
	--remove-orphans \
	--renew-anon-volumes \
	--build
sleep 1s

echo -e "...done relaunching compose services."
