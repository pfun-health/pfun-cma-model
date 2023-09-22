#!/usr/bin/env bash

ROOT_DIR=${ROOT_DIR:-$(git rev-parse --show-toplevel)}

# sample local deployment
base_url=http://localhost:8000/ ${ROOT_DIR}/runtime/scripts/sample-endpoint.sh
