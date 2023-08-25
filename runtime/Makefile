deploy: tests deploy_dependencies build_venv prepare_deployment deploy_chalice test_deployment

deploy_dependencies:
	echo "deploying dependencies layer..."
	cd ${HOME}/Git/pfun-cma-model-deps && ./deploy.sh
	sleep 2s

deploy_local: build_venv tests prepare_deployment
	@sleep 1s;
	@/tmp/venv310/bin/chalice local
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
	/tmp/venv310/bin/python "${HOME}/Git/pfun-cma-model/scripts/fix-lambda-layer-version.py"
	sleep 1s

deploy_chalice:
	echo 'deploying...'
	sleep 1s
	/tmp/venv310/bin/chalice deploy
	sleep 1s

test_deployment:
	echo 'testing...'
	sleep 1s
	(/tmp/venv310/bin/chalice invoke -n run_model | jq) && \
		echo 'Success (see above)...done testing lambda function run_model.'
	sleep 1s
	(./scripts/sample-endpoint.sh) && \
		echo 'Success (see above)...done testing HTTP endpoint.'

activate:
	sleep 1s
	source /tmp/venv310/bin/activate
	sleep 1s
	echo -e "...activated the venv310.\nPython version: '$(python --version)'"

tests:
	sleep 1s
	pytest

tests-interactive:
	sleep 1s
	pytest --pdb

plot-interactive:
	sleep 1s
	ipython -i ${HOME}/Git/pfun-cma-model/scripts/plot-interactive.py

clean:
	rm -rf ${HOME}/Git/pfun-cma-model/.chalice/deployments/*
	rm -rf ${HOME}/Git/pfun-cma-model/__pycache__
	rm -rf /tmp/venv310
	rm -rf ${HOME}/Git/pfun-cma-model/chalicelib/www/lib
	echo '...cleaned the deployments and venv.'

docs:
	pdoc --html .

generate-sdk:
	echo 'generating SDK...'
	sleep 1s
	/tmp/venv310/bin/chalice generate-sdk --sdk-type javascript ./chalicelib/www
	sleep 1s
	echo '...generated SDK'