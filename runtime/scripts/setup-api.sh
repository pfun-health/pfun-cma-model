#!/usr/bin/env bash

# Create a usage plan for the pfun-cma-model API
source ./.envrc

set -e

unset AWS_ENDPOINT_URL
API_NAME="PFun\ CMA\ Model\ Backend"
API_STAGE="api"
API_KEY_NAME="PFun\ CMA\ Model\ API\ Key"
AWS_REGION="us-east-1"

echo 'getting API ID...'
API_ID=$(aws apigateway get-rest-apis --query "items[?name=='${API_NAME//\\/}'].id" --output text)
sleep 1s
echo "API ID: ${API_ID}"

echo 'creating usage plan...'
USAGE_PLAN_ID=$(
	aws apigateway create-usage-plan \
		--name pfun-cma-model-dev-plan \
		--description 'for use with development of the pfun-cma-model API' \
		--api-stages "apiId=${API_ID},stage=${API_STAGE}" |
		jq '.id'
)
sleep 1s
echo "Usage Plan ID: ${USAGE_PLAN_ID}"

echo 'creating API key...'
API_KEY_ID=$(aws apigateway create-api-key \
	--name "${API_KEY_NAME//\\/}" \
	--enabled \
	--generate-distinct-id |
	jq '.id')
sleep 1s
echo "API Key ID: ${API_KEY_ID}"

echo 'creating usage plan key...'
aws apigateway create-usage-plan-key \
	--usage-plan-id ${USAGE_PLAN_ID} \
	--key-id ${API_KEY_ID} \
	--key-type "API_KEY"
sleep 1s

echo 'creating deployment...'
function get-resources() {
	aws apigateway get-resources \
		--rest-api-id ukpfvcn2ac |
		jq | python -c \
		$'
    import json
    import sys
    d = json.loads(sys.stdin.read())
    print(
        " ".join([item["id"] for item in d["items"] if
                  item["path"] in ["/", "/fit", "/run", "/run-at-time",]])
    )'
}

# Set the API key for each resource
for RESOURCE_ID in $(get-resources); do
	aws apigateway update-api-key \
		--api-key ${API_KEY_ID} \
		--patch-operations \
		op='replace',path=/${RESOURCE_ID}/stageKeys,value=${RESOURCE_ID}/${API_STAGE}
done

aws apigateway update-deployment \
	--rest-api-id ${API_ID} \
	--stage-name ${API_STAGE}
