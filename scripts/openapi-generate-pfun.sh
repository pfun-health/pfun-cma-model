#!/usr/bin/env bash

echo -e "generating openapi client for pfun-cma-model..."

OPENAPI_URI="https://pfun-cma-model-446025415469.us-central1.run.app/openapi.json"

docker run --rm -v "${PWD}:/local" openapitools/openapi-generator-cli generate \
    -i "${OPENAPI_URI}" \
    -g python \
    -o /local/generated_clients/pfun-cma-model-client

sleep 1s;

