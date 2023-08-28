#!/usr/bin/env bash

echo 'deleting old venv'
set +e
rm -rf /tmp/venv310
set -e

echo 'creating new venv'
# ! important (virtual environment)
$HOME/.pyenv/versions/3.10.9/bin/python -m venv /tmp/venv310

set +e
# ! deactivate pyenv just in case
pyenv deactivate
set -e

sleep 1s
echo 'activate venv'
sleep 1s

source /tmp/venv310/bin/activate
sleep 1s
echo -e "** python version: **\n$(which python)\n"
sleep 1s

echo 'installing from venv...'
TMPDIR=/tmp/venv310/lib/python3.10/site-packages
/tmp/venv310/bin/python -m pip install \
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
echo "installling minpack..."
./scripts/install_minpack.sh
sleep 1s

# set +e
# temp_dir=${TMPDIR}

# # List of patterns to remove
# patterns=(
# 	"*.dist-info"
# 	"*.egg-info"
# 	"*.pyc"
# 	"__pycache__"
# 	"tests"
# 	"doc"
# 	"datasets"
# )

# # Loop through the patterns and remove matching files
# for pattern in "${patterns[@]}"; do
# 	find "${temp_dir}/" -name "${pattern}" -exec rm -rf {} \;
# done

# echo "...done cleaning up dependencies."
# set -e
# sleep 1s
