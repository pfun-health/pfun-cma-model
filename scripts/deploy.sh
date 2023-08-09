#!/bin/sh

echo 'deleting old venv'
set +e
rm -rf /tmp/venv310
set -e

echo 'creating new venv'
# ! important (virtual environment)
python3.10 -m venv /tmp/venv310

echo 'activate venv'
. /tmp/venv310/bin/activate

echo 'installing from venv...'
TMPDIR=/tmp/venv310/lib/python3.10/site-packages
python3.10 -m pip install \
	--upgrade \
	--platform manylinux2014_x86_64 \
	--target $TMPDIR/ \
	--implementation cp \
	--python-version 3.10 \
	--only-binary=:all: \
	--no-cache-dir \
	-r requirements.txt \
	-r requirements-dev.txt
sleep 1s

echo 'cleaning up...'
# cleanup
mv $TMPDIR/bin/* /tmp/venv310/bin/
sleep 1s

# install minpack
./scripts/install_minpack.sh
sleep 1s;

set +e
find $TMPDIR/ -name "*.dist-info" -exec rm -rf {} \;
find $TMPDIR/ -name "*.egg-info" -exec rm -rf {} \;
find $TMPDIR/ -name "*.pyc" -exec rm -rf {} \;
find $TMPDIR/ -name "__pycache__" -exec rm -rf {} \;
find $TMPDIR/ -name "tests" -exec rm -rf {} \;
find $TMPDIR/ -name "doc" -exec rm -rf {} \;
find $TMPDIR/ -name "datasets" -exec rm -rf {} \;
echo "...done cleaning up dependencies."
set -e
sleep 1s

sleep 1s
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
