#!/usr/bin/env sh

LLVM_CONFIG=/usr/bin/llvm-config-14 \
    poetry run pip wheel --no-cache-dir --use-pep517 "llvmlite (==0.41.1)"
