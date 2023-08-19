deploy: deploy_dependencies build_venv prepare_deployment deploy_chalice test_deployment

deploy_dependencies:
	echo "deploying dependencies layer..."
	cd ${HOME}/Git/pfun-cma-model-deps && ./deploy.sh
	cd -
	sleep 2s

deploy_local: build_venv prepare_deployment
	/tmp/venv310/bin/chalice local &
	sleep 1s;
	echo -e '...Success (see above)...deployed locally.'

deploy_model: build_venv prepare_deployment deploy_chalice test_deployment
	echo -e '...Success (see above)...done deploying.'

build_venv:
	./scripts/create-venv.sh
	sleep 1s

prepare_deployment:
	set +e
	rm -rf ${HOME}/Git/pfun-cma-model/.chalice/deployments/*
	set -e
	echo 'preparing for deployment...'
	sleep 1s
	/tmp/venv310/bin/python "${HOME}/Git/pfun-cma-model/scripts/fix-lambda-layer-version.py"
	sleep 1s

deploy_chalice:
	echo 'deploying...'
	sleep 1s
	/tmp/venv310/bin/chalice deploy

test_deployment:
	echo 'testing...'
	sleep 1s
	(/tmp/venv310/bin/chalice invoke -n run_model | jq) && \
		echo 'Success (see above)...done testing.'

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
	echo '...cleaned the deployments and venv.'