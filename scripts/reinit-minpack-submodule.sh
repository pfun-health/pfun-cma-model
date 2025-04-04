#!/usr/bin/env bash

# This script is used to reinitialize the minpack submodule.

git submodule add --force ./minpack ./minpack && \
    git submodule init && \
        git submodule update
