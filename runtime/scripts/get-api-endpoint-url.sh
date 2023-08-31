#!/bin/bash

export api_id=$(aws apigateway get-rest-apis --query "items[?name=='PFun CMA Model Backend'].id" --output text)
export api_endpoint_url="https://${api_id}.execute-api.us-east-1.amazonaws.com/api"
export run_resource_id=$(aws apigateway get-resources --rest-api-id $(aws apigateway get-rest-apis --query "items[?name=='PFun CMA Model Backend'].id" --output text) --query "items[?pathPart=='run'].id" --output text)

export API_KEY_ID=$(aws apigateway get-api-keys --query "items[?name=='pfun-cma-model-dev-client'].id" --output text)
echo "API KEY ID: ${API_KEY_ID}"
echo "endpoint URL: $api_endpoint_url"
echo "API ID: $api_id"
echo "/run Resource ID: $run_resource_id"
