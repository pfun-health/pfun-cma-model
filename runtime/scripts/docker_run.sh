#!/usr/bin/env bash

# version for pfun-cma-model

docker run -it \
       --name pfun-cma-model \
       -p 127.0.0.1:8000:8000/tcp \
       --rm \
       rocapp/pfun-cma-model:latest