#!/usr/bin/env bash

docker compose exec pfun-cma-model \
    bash -c 'cd /app && uv version'
