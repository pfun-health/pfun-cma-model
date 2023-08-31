#!/bin/bash

source ${HOME}/Git/pfun-cma-model/.envrc

unset AWS_ENDPOINT_URL
proxy_secret=$(aws secretsmanager get-secret-value --secret-id pfun-cma-model-rapid-api-proxy-secret --region us-east-1 | jq -r '.SecretString')
api_key=$(aws secretsmanager get-secret-value --secret-id pfun-cma-model-rapidapi-key --region us-east-1 | jq -r '.SecretString')

sleep 0.1s

endpoint=${1:-run}

extra_params=${2:-}

if [ "$endpoint" == "sdk" ]; then
	extra_params="-O"
fi

base_url=${base_url:-https://$(aws apigateway get-rest-apis --query "items[?name=='PFun CMA Model Backend'].id" --output text).execute-api.us-east-1.amazonaws.com/api/}
url="${base_url}${endpoint}"
echo "URL=${url}"

CMD_ARGS=$(
	cat <<EOF
curl \
--url "${url}" \
--header "X-RapidAPI-Proxy-Secret: $proxy_secret" \
--header "X-RapidAPI-Host: pfun-cma-model-api.p.rapidapi.com" \
--header "X-RapidAPI-Key: $api_key" \
--header "Authorization: Bearer allow" \
${extra_params}
EOF
)

echo -e "command:\n${CMD_ARGS}\n"

eval "${CMD_ARGS}"

sleep 0.1s

if [ "$endpoint" == "sdk" ]; then
	python $HOME/Git/pfun-cma-model/runtime/scripts/extract-sdk.py || exit 1
	sleep 0.1s
	echo "...updated SDK"
fi
