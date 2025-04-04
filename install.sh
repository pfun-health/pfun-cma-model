#!/usr/bin/env bash

# install.sh

source ./.venv/bin/activate

# install dependencies

# build the package (./dist/...)
rm -r ./dist || true
rm -rf ./build || true
rm -rf ./minpack || true
python -m pip uninstall --yes minpack pfun-cma-model

if [[ $(which poetry) ]]; then
    echo "Poetry is installed" && \
        poetry run build-minpack && \
            poetry build --no-cache && \
                poetry run python -m pip install --upgrade dist/*.whl
else
    echo "Poetry is not installed."
    python ./scripts/build_minpack.py && \
        python -m pip install --upgrade build && \
            python -m build --no-cache && \
                python -m pip install --upgrade dist/*.whl
fi

# install the package from the wheel (./dist/*.whl)
