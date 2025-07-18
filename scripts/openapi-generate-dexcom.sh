#!/usr/bin/env bash

# scripts/openapi-generate-dexcom.sh

echo -e "generating openapi client for dexcom..."

OPENAPI_URI="https://raw.githubusercontent.com/pfun-health/dexcom-openapi-schema/refs/heads/main/openapi.json"
# python python-fastapi typescript-jquery dynamic-html html html2
CLIENT_TYPE="${1:-dynamic-html}"

echo -e "Client Type: '$CLIENT_TYPE'..." && \
    sleep 2s

docker run --rm -v "${PWD}:/local" openapitools/openapi-generator-cli generate \
    -i "${OPENAPI_URI}" \
    -g "${CLIENT_TYPE}" \
    --skip-validate-spec \
    -o "/local/generated_clients/${CLIENT_TYPE}-dexcom-client"

sleep 1s

sudo chown -R $USER:users ./generated_clients
