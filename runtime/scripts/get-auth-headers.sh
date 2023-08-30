#!/bin/bash
proxy_secret=$(aws secretsmanager get-secret-value --secret-id pfun-cma-model-rapid-api-proxy-secret --region us-east-1 | jq -r '.SecretString')
api_key=$(aws secretsmanager get-secret-value --secret-id pfun-cma-model-rapidapi-key --region us-east-1 | jq -r '.SecretString')

export AUTH_HEADERS="$(
    cat <<EOM
{
    "X-RapidAPI-Proxy-Secret": "$proxy_secret",
    "X-RapidAPI-Host": "pfun-cma-model-api.p.rapidapi.com",
    "X-RapidAPI-Key": "$api_key",
    "Content-Type": "application/json",
    "Authorization": "Bearer allow"
}
EOM
)" >headers.json

echo "${AUTH_HEADERS}" | jq
