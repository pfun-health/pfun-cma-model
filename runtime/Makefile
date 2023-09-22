deploy: tests deploy-chalice test-deployment

STAGE ?= dev
# Variables
DOMAIN_NAME = api.$(STAGE).pfun.app
APP ?= pfun-cma-model
LOG_GROUP_NAME = /aws/lambda/$(APP)-$(STAGE)

local:
	@echo "running locally with STAGE=$(STAGE)..."
	sleep 1s
	@cd ${HOME}/Git/pfun-cma-model/runtime && chalice local --stage $(STAGE)
	sleep 2s

deploy-dependencies:
	echo "deploying dependencies layer..."
	@cd ${HOME}/Git/pfun-cma-model-deps && chalice deploy --stage ${STAGE}
	sleep 2s

deploy-local: tests
	@sleep 1s;
	@chalice local
	@sleep 1s;
	@echo -e '...Success (see above)...deployed locally.'
	@sleep 1s;
	@make test_local_deployment
	sleep 1s;

test-local:
	sleep 1s;
	echo -e '...Testing local deployment...';
	base_url='http://localhost:8000/' \
	${HOME}/Git/pfun-cma-model/runtime/scripts/sample-local-deployment.sh
	sleep 1s
	@echo '...Success (see above)...done testing.'

deploy-chalice: generate-sdk
	@echo 'deploying...'
	@chalice deploy --stage $(STAGE)
	sleep 1s
	@echo '...deployed with latest sdk.'

test-deployment:
	@echo 'testing...'
	sleep 1s
	(${HOME}/Git/pfun-cma-model/runtime/scripts/sample-endpoint.sh) && \
		echo 'Success (see above)...done testing HTTP endpoint.'

tests:
	sleep 1s
	pytest

tests-interactive:
	sleep 1s
	pytest --pdb

plot-interactive:
	sleep 1s
	ipython -i ${HOME}/Git/pfun-cma-model/runtime/scripts/plot-interactive.py

clean:
	rm -rf ${HOME}/Git/pfun-cma-model/runtime/.chalice/deployments/*
	rm -rf ${HOME}/Git/pfun-cma-model/__pycache__
	rm -rf ${HOME}/Git/pfun-cma-model/runtime/chalicelib/www/lib
	echo '...cleaned the deployments and venv.'

docs:
	pdoc --html .

generate-sdk:
	echo 'running chalice locally...'
	sh -c 'chalice local &'
	echo 'generating SDK...'
	sleep 1s
	${HOME}/Git/pfun-cma-model/runtime/scripts/generate-sdk.sh
	sleep 1s
	echo '...generated SDK'
	pkill chalice

create-role:
	@aws iam create-role --role-name pfun-cma-model-$(STAGE) --assume-role-policy-document file://./.chalice/trust-policy.json
	@echo '...done with setup of role.'

setup-role-policy:
	@aws iam attach-role-policy --role-name pfun-cma-model-$(STAGE) --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
	@aws iam attach-role-policy --role-name pfun-cma-model-$(STAGE)-api_handler --policy-arn arn:aws:iam::aws:policy/AmazonAPIGatewayInvokeFullAccess
	@aws iam attach-role-policy --role-name pfun-cma-model-$(STAGE) --policy-arn arn:aws:iam::aws:policy/AmazonAPIGatewayInvokeFullAccess
	@echo '...done with setup of role policy.'

create-hosted-zone:
	@aws route53 create-hosted-zone --name api.$(STAGE).pfun.app --caller-reference 12345

create-acm-cert:
	@aws acm request-certificate --domain-name "*.api.$(STAGE).pfun.app" \
    --validation-method DNS --idempotency-token 12345 \
    --options CertificateTransparencyLoggingPreference=DISABLED

generate-swagger:
	sh -c 'cd ${HOME}/Git/pfun-cma-model/runtime/chalicelib/www && aws apigateway get-export --rest-api-id oias8ms59c --stage-name api --export-type swagger --parameters extensions='apigateway' swagger.json --output json && cd -'
	echo '...generated swagger.json'

clear_logs:
	# Delete all log streams in the log group
	# aws logs describe-log-streams --log-group-name $(LOG_GROUP_NAME) --query 'logStreams[*].logStreamName' --output text | xargs -I {} aws logs delete-log-stream --log-group-name $(LOG_GROUP_NAME) --log-stream-name {}
	# If you want to delete the log group entirely, uncomment the following line
	@aws logs delete-log-group --log-group-name $(LOG_GROUP_NAME)

SOURCE_BUCKET = pfun-cma-model-www
NEW_BUCKET = pfun-cma-model-www-$(STAGE)
REGION = us-east-1

copy-bucket:
	aws s3api create-bucket --bucket $(NEW_BUCKET) --region $(REGION)
	aws s3 sync s3://$(SOURCE_BUCKET) s3://$(NEW_BUCKET)
	-aws s3api get-bucket-policy --bucket $(SOURCE_BUCKET) --query 'Policy' --output text > bucket-policy.json && \
	aws s3api put-bucket-policy --bucket $(NEW_BUCKET) --policy file://bucket-policy.json || true
	aws s3api get-bucket-acl --bucket $(SOURCE_BUCKET) > bucket-acl.json || true && aws s3api put-bucket-acl --bucket $(NEW_BUCKET) --access-control-policy file://bucket-acl.json || true
	echo '...copied bucket'