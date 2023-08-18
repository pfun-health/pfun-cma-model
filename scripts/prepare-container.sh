#!/bin/bash

# prepare-container.sh

# prepare the dev container for pfun-cma-model
set -e

./scripts/create-venv.sh || exit 1

echo -e '...successfully created venv!'

conda install -r requirements-dev.txt -r requirements.txt

echo -e '...successfully installed dependencies!'

echo -e '...done.'

exit 0
