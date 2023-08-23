#!/bin/bash

source ${HOME}/Git/pfun-cma-model/.envrc

proxy_secret=$(aws secretsmanager get-secret-value --secret-id pfun-cma-model-rapid-api-proxy-secret --region us-east-1 | jq -r '.SecretString')
api_key=$(aws secretsmanager get-secret-value --secret-id pfun-cma-model-rapidapi-key --region us-east-1 | jq -r '.SecretString')
sleep 1s

endpoint=${1:-run}
set +e
default_url=$(chalice url)
set -e
url="${base_url}${endpoint}"
echo "URL=${url}"
base_url=${base_url:-${default_url}}

curl --request POST --url ${url} \
	--header "X-RapidAPI-Proxy-Secret: $proxy_secret" \
	--header "X-RapidAPI-Key: $api_key" \
	--header "Content-Type: application/json" \
	--header "Authorization: Bearer allow" \
	--data '{}'
sleep 1s
