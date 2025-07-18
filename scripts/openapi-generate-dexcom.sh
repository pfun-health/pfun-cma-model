#!/usr/bin/env bash

# scripts/openapi-generate-dexcom.sh

echo -e "generating openapi client for dexcom..."

OPENAPI_URI="https://raw.githubusercontent.com/pfun-health/dexcom-openapi-schema/refs/heads/main/openapi.json"

docker run --rm -v "${PWD}:/local" openapitools/openapi-generator-cli generate \
    -i "${OPENAPI_URI}" \
    -g python-fastapi \
    --skip-validate-spec \
    -o /local/generated_clients/fastapi-dexcom-client

sleep 1s
