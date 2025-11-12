#!/usr/bin/env bash

set -e

echo -e "generating openapi client for pfun-cma-model..."

OPENAPI_URI="https://pfun-cma-model-446025415469.us-central1.run.app/openapi.json"
OPENAPI_JSON="${PWD}/openapi.json"
OUTPUT_DIR="${PWD}/generated_clients/pfun-cma-model-client"

download_openapi_json() {
    echo -e "\nDownloading openapi json..."
    curl -o "${OPENAPI_JSON}" "${OPENAPI_URI}"
    sleep 1s;
}

# download openapi json
download_openapi_json

# use docker if available
if [ $(which docker) ]; then
    echo -e "using docker..."
    docker run --rm -v "${PWD}:/local" openapitools/openapi-generator-cli generate \
        -i "${OPENAPI_JSON}" \
        -g python \
        -o "/local/${OUTPUT_DIR}"
    sleep 1s;
    exit 0
fi

if [ $(which uv) ]; then
    echo -e "using uv..."
     if [ "$(which openapi-generator-cli)" == '' ]; then
         uv add --dev 'openapi-generator-cli[jdk4py]'
     fi
     openapi-generator-cli generate \
         -i "${OPENAPI_JSON}" \
         -g python \
         -o "${OUTPUT_DIR}"
    sleep 1s;
    exit 0
fi
