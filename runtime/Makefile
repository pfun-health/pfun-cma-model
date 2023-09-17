deploy: tests deploy-chalice test-deployment

STAGE ?= dev

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

test-local-deployment:
	sleep 1s;
	echo -e '...Testing local deployment...';
	base_url='http://localhost:8000/' \
	${HOME}/Git/pfun-cma-model/scripts/sample-local-deployment.sh
	sleep 1s
	@echo '...Success (see above)...done testing.'

deploy-localstack: 
	@echo "Deploying to localstack..."
	@echo "AWS_ENDPOINT_URL: ${AWSLOCALSTACK_ENDPOINT_URL}"
	@echo "AWS_PROFILE: localstack"
	@bash -c "cd ${HOME}/Git/pfun-cma-model/runtime && \
		AWS_ENDPOINT_URL=${AWSLOCALSTACK_ENDPOINT_URL} \
		AWS_PROFILE=localstack chalice-local deploy"
	@echo "...Done."

deploy-chalice: generate-sdk
	@echo 'deploying...'
	@chalice deploy --stage $(STAGE)
	sleep 1s
	@echo '...deployed with latest sdk.'

test-deployment:
	@echo 'testing...'
	sleep 1s
	(./scripts/sample-endpoint.sh) && \
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
	@aws iam create-role --role-name pfun-cma-model-dev --assume-role-policy-document file://./.chalice/trust-policy.json
	@echo '...done with setup of role.'

setup-role-policy:
	@aws iam attach-role-policy --role-name pfun-cma-model-dev --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
	@aws iam attach-role-policy --role-name pfun-cma-model-dev-api_handler --policy-arn arn:aws:iam::aws:policy/AmazonAPIGatewayInvokeFullAccess
	@aws iam attach-role-policy --role-name pfun-cma-model-dev --policy-arn arn:aws:iam::aws:policy/AmazonAPIGatewayInvokeFullAccess
	@echo '...done with setup of role policy.'

generate-swagger:
	sh -c 'cd ${HOME}/Git/pfun-cma-model/runtime/chalicelib/www && aws apigateway get-export --rest-api-id oias8ms59c --stage-name api --export-type swagger --parameters extensions='apigateway' swagger.json --output json && cd -'
	echo '...generated swagger.json'