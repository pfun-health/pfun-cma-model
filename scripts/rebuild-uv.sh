#!/usr/bin/env bash

set -e

# scripts/rebuild-uv.sh : Script to perform a local rebuild of the uv environment.

# Load common functions
source "$(dirname "$0")/_funcs.def.sh"

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

echo "...local rebuild using uv version ${NEW_VERSION}" &&
	uv version "${NEW_VERSION}" &&
	echo -e "updated version" &&
	full_uv_sync &&
	uv build &&
	echo -e "...rebuilt locally (uv version ${NEW_VERSION})."

