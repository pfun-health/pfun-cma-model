#!/usr/bin/env bash

# install.sh

setup-venv() {
    # This script is used to install the package in a virtual environment.
    if [ -d ".venv" ]; then
        echo "Virtual environment already exists."
    else
        echo "Creating virtual environment..."
        python3 -m venv ./.venv
    fi
    # Activate the virtual environment
    source ./.venv/bin/activate
}


cleanup-build() {
    # cleanup the existing build
    rm -r ./dist >/dev/null 2>&1 || true
    rm -rf ./build >/dev/null 2>&1 || true
    rm -rf ./minpack >/dev/null 2>&1 || true
    python -m pip uninstall --yes minpack pfun-cma-model >/dev/null 2>&1 || true
}


install-package() {
    # install the package from the source
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
                    python -m pip install --no-cache --upgrade dist/*.whl
    fi
}


##
# Main script execution
##

setup-venv
cleanup-build 
install-package