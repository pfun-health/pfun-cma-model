#!/bin/sh

echo "deploying dependencies layer..."
cd $HOME/Git/pfun-cma-model-deps &&
	./deploy.sh
sleep 2s

./scripts/create-venv.sh

set +e
rm $HOME/Git/pfun-cma-model/.chalice/deployments/*
set -e

echo 'deploying...'
sleep 1s
/tmp/venv310/bin/chalice deploy || exit

sleep 1s
echo 'testing...'
# test the endpoint
(/tmp/venv310/bin/chalice invoke -n run_model | jq) &&
	echo 'Success (see above)...done testing.' ||
	echo "Failure (see above)...done testing."

sleep 1s
deactivate
