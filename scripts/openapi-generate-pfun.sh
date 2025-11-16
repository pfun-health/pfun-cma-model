#!/usr/bin/env bash
# openapi-generate-pfun.sh : generate openapi client for pfun-cma-model API

set -e

echo -e "generating openapi client for pfun-cma-model..."

OPENAPI_URI="https://pfun-cma-model-446025415469.us-central1.run.app/openapi.json"
OPENAPI_JSON="${PWD}/openapi.json"
OUTPUT_DIR="${PWD}/generated_clients/pfun-cma-model-client"

export OPENAPI_GENERATE_ARGS OPENAPI_GENERATE_CMD
CLIENT_TYPE="python"
OPENAPI_GENERATE_ARGS="-i ${OPENAPI_JSON} -g ${CLIENT_TYPE} -o ${OUTPUT_DIR}"
OPENAPI_GENERATE_ARGS="${OPENAPI_GENERATE_ARGS} --git-user-id=pfun-health --git-repo-id=pfun-cma-model"
OPENAPI_GENERATE_CMD="openapi-generator-cli generate ${OPENAPI_GENERATE_ARGS}"

download_openapi_json() {
    echo -e "\nDownloading openapi json..."
    curl -o "${OPENAPI_JSON}" "${OPENAPI_URI}"
    sleep 1s;
}

# download openapi json
download_openapi_json

replace_server_host() {
    echo -e "\nReplacing server host (fixing in generated client README and client code)..."
    sed -i "s/http:\/\/localhost/https:\/\/pfun-cma-model-446025415469.us-central1.run.app/g" "${OUTPUT_DIR}/README.md"
    find "${OUTPUT_DIR}" -type f -name "*.py" -exec sed -i "s/http:\/\/localhost/https:\/\/pfun-cma-model-446025415469.us-central1.run.app/g" {} \;
    sleep 1s;
}

copy_to_beetus() {
    BEETUS_CLIENT_DIR="$(realpath ../beetus/pfun_cma_model_client)"
    echo -e "\nCopying to beetus (${BEETUS_CLIENT_DIR})..."
    rm -rf "${BEETUS_CLIENT_DIR}"
    cp -r --interactive "${OUTPUT_DIR}" "${BEETUS_CLIENT_DIR}"
    sleep 1s;
}

# # use docker if available
# if [ $(which docker) ]; then
#     echo -e "using docker..."
#     docker run --rm -v "${PWD}:${PWD}" \
#         openapitools/${OPENAPI_GENERATE_CMD}
#     sleep 1s;
#     # posthoc actions
#     replace_server_host && copy_to_beetus
#     exit 0
# fi

if [ $(which uv) ]; then
    echo -e "using uv..."
    if [ "$(which openapi-generator-cli)" == '' ]; then
        uv add --dev 'openapi-generator-cli[jdk4py]'
    fi
    bash -c "${OPENAPI_GENERATE_CMD}"
    sleep 1s;
    # posthoc actions
    replace_server_host && copy_to_beetus
    exit 0
fi
