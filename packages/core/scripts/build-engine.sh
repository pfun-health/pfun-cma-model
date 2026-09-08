#!/usr/bin/env bash

set -e
bash -c 'git clone https://github.com/pfun-health/pfun-cma-engine-c.git && cd pfun-cma-engine-c && make clean && make'
sleep 1s
echo -e "...done building pfun-cma-engine-c (packages/core)."
