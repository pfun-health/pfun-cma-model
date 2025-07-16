#!/usr/bin/env bash

set -e


setup-bash-completion() {
    echo -e "Setting up bash completion for pfun-cma-model..."
    _PFUN_CMA_MODEL_COMPLETE=bash_source pfun-cma-model \
        > ~/.pfun-cma-model-complete.bash
}


setup-bash-completion