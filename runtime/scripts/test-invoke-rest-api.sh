#!/bin/bash

source ./scripts/get-auth-headers.sh
source ./scripts/get-api-endpoint-url.sh

authorizer_id=v0opmq
aws apigateway test-invoke-authorizer \
    --rest-api-id "${api_id}" \
    --authorizer-id "${authorizer_id}" \
    --headers "${AUTH_HEADERS}"

aws apigateway test-invoke-method --http-method GET \
    --headers "${AUTH_HEADERS}" \
    --rest-api-id "${api_id}" \
    --resource-id "${run_resource_id}"
