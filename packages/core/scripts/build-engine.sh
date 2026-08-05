#!/usr/bin/env bash

set -e

cd ../pfun-cma-engine-c
mkdir -p build && cd build
cmake ..
make
