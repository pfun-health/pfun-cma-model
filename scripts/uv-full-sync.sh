#!/usr/bin/env bash

set -e

# Load common functions
source "$(dirname "$0")/_funcs.def.sh"

# execute full sync (for prod/pre-prod deployments)

full_uv_sync