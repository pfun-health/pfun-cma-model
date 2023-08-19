#!/bin/bash
source ${HOME}/Git/pfun-cma-model/.envrc
proxy_secret=$(aws secretsmanager get-secret-value --secret-id pfun-cma-model-rapid-api-proxy-secret --region us-east-1 | jq -r '.SecretString')
api_key=$(aws secretsmanager get-secret-value --secret-id pfun-cma-model-rapidapi-key --region us-east-1 | jq -r '.SecretString')
curl --request POST --url http://"$(chalice url)" \
	--header "X-RapidAPI-Proxy-Secret: $proxy_secret" \
	--header "X-RapidAPI-Key: $api_key" \
	--header "Content-Type: application/json" \
	--header "Authorization: Bearer allow" \
	--data '{}'
