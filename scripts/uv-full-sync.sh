#!/usr/bin/env sh

# scripts/uv-full-sync.sh
# execute full sync (for prod/pre-prod deployments)

set -e

# Load common functions
. "$(dirname "$0")/_funcs.def.sh"

# remove old virtual environment, then fully sync
rm -rf ./.venv ./packages/pfun_common/.venv

if [ "$1" = '-gui' ]; then
    full_uv_sync_qt_gui
else
    full_uv_sync
fi
