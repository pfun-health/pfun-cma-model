#!/bin/bash

source ./scripts/get-auth-headers.sh
source ./scripts/get-api-endpoint-url.sh

aws apigateway test-invoke-method --http-method POST \
    --headers "${AUTH_HEADERS}" \
    --rest-api-id "${api_id}" \
    --resource-id "${run_resource_id}"
