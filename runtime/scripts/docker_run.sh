#!/usr/bin/env bash

# version for pfun-cma-model

docker run --rm \
       --name pfun-cma-model \
       -p 0.0.0.0:8003:8001/tcp \
       -p 0.0.0.0:8002:8002/tcp \
       -it \
       rocapp/pfun-cma-model:latest
