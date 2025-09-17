#!/usr/bin/env bash

# new-version.sh
# Bump pfun-cma-model to a new patch version, record it.

set -e

# bump pfun-cma-model package version
uv version --bump patch && \
    uv sync && \
    uv build

# build and start the services in the background
nohup docker compose up -d --build & > ./logs/docker-compose.log 2>&1 &