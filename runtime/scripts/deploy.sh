#!/bin/sh

echo "deploying dependencies layer..."
cd $HOME/Git/pfun-cma-model-deps && ./deploy.sh || exit 1
cd - || exit 1
sleep 2s

./scripts/create-venv.sh

set +e
rm $HOME/Git/pfun-cma-model/.chalice/deployments/*
set -e

echo 'preparing for deployment...'
sleep 1s
/tmp/venv310/bin/python "${HOME}/Git/pfun-cma-model/scripts/fix-lambda-layer-version.py" || exit 1
sleep 1s

echo 'deploying...'
sleep 1s
/tmp/venv310/bin/chalice deploy || exit

sleep 1s
echo 'testing...'
sleep 1s
# test the endpoint
(/tmp/venv310/bin/chalice invoke -n run_model | jq) &&
	echo 'Success (see above)...done testing.' ||
	echo "Failure (see above)...done testing."

sleep 1s
deactivate
