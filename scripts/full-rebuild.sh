#!/usr/bin/env bash

set -e

# scripts/full-rebuild.sh : Script to perform a full rebuild of the uv environment and relaunch docker compose services.

# Load common functions
source "$(dirname "$0")/_funcs.def.sh"

docker compose down

export CURRENT_VERSION NEW_VERSION

get_version() {
	uv version | python -c 'import sys; print(sys.stdin.read().split(" ")[-1].strip())'
}
CURRENT_VERSION="$(get_version)"
echo -e "current version: '$CURRENT_VERSION'"
sleep 1s

NEW_VERSION="${1:-$CURRENT_VERSION}"
echo -e "updating to latest: '$NEW_VERSION' ..."
sleep 1s

echo "..." &&
	uv version "${NEW_VERSION}" &&
	echo -e "updated version" &&
	full_uv_sync &&
	uv build &&
	echo -e "...rebuilt locally."

docker compose up -d \
	--remove-orphans \
	--renew-anon-volumes \
	--build
sleep 1s

echo -e "...done relaunching compose services."
