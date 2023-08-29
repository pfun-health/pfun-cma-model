#!/bin/bash

api_id=$(aws apigateway get-rest-apis --query "items[?name=='PFun CMA Model Backend'].id" --output text)
api_endpoint_url="https://${api_id}.execute-api.us-east-1.amazonaws.com/api"

echo $api_endpoint_url
