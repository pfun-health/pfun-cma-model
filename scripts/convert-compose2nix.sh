#!/usr/bin/env sh

# scripts/convert-compose2nix.sh
# convert docker-compose.yml to docker-compose.nix

set -e

nix run github:aksiksi/compose2nix -- \
    -project=pfun-cma-model \
    -include_env_files \
    -env_files '.env' \
    -output 'oci-container.nix'
