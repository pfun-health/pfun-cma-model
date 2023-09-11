deploy: tests deploy_dependencies build_venv prepare_deployment deploy_chalice test_deployment

deploy_dependencies:
	echo "deploying dependencies layer..."
	cd ${HOME}/Git/pfun-cma-model-deps && chalice deploy
	sleep 2s

deploy_local: build_venv tests prepare_deployment
	@sleep 1s;
	@chalice local
	@sleep 1s;
	@echo -e '...Success (see above)...deployed locally.'
	@sleep 1s;
	@make test_local_deployment
	sleep 1s;

test_local_deployment:
	sleep 1s;
	echo -e '...Testing local deployment...';
	base_url='http://localhost:8000/' \
	${HOME}/Git/pfun-cma-model/scripts/sample-local-deployment.sh
	sleep 1s
	echo '...Success (see above)...done testing.'

deploy_model: build_venv generate-sdk prepare_deployment deploy_chalice test_deployment
	echo -e '...Success (see above)...done deploying.'

build_venv:
	./scripts/create-venv.sh
	sleep 1s

prepare_deployment:
	echo 'preparing for deployment...'
	echo 'generating openapi.json...'
	chalice generate-models > ./chalicelib/openapi.json
	echo '...generated openapi.json'
	echo 'cleaning old deployments...'
	set +e
	rm -rf ${HOME}/Git/pfun-cma-model/.chalice/deployments/*
	set -e
	sleep 1s
	echo '...cleared old deployments'
	sleep 1s
	echo 'fixing lambda layer version...'
	sleep 1s
	python "${HOME}/Git/pfun-cma-model/scripts/fix-lambda-layer-version.py"
	sleep 1s

deploy_chalice:
	echo 'deploying...'
	sleep 1s
	chalice deploy
	sleep 1s
	echo '...deployed.'
	sleep 1s
	echo 'generate sdk...'
	@make generate-sdk
	sleep 1s
	echo '...generated sdk.'
	sleep 1s
	echo 'deploy with latest sdk...'
	chalice deploy
	sleep 1s
	echo '...deployed with latest sdk.'

test_deployment:
	echo 'testing...'
	sleep 1s
	(chalice invoke -n run_model | jq) && \
		echo 'Success (see above)...done testing lambda function run_model.'
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
	echo 'generating SDK...'
	sleep 1s
	${HOME}/Git/pfun-cma-model/runtime/scripts/generate-sdk.sh
	sleep 1s
	echo '...generated SDK'

create-role:
	@aws iam create-role --role-name pfun-cma-model-dev --assume-role-policy-document file://./.chalice/trust-policy.json
	@echo '...done with setup of role.'

setup-role-policy:
	@aws iam attach-role-policy --role-name pfun-cma-model-dev --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
	@echo '...done with setup of role policy.'
