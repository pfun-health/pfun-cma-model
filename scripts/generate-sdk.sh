#!/bin/bash

echo 'generating sdk...'

export ROOT_REPO_DIR=$HOME/Git/pfun-cma-model
export base_url=${base_url:-localhost:8000/}

$ROOT_REPO_DIR/runtime/scripts/sample-endpoint.sh sdk || exit
sleep 0.1s

cd $ROOT_REPO_DIR/runtime &&
    chalice generate-models >./chalicelib/www/openapi.json || exit &&
    cd - || exit
sleep 0.1s

aws s3 cp \
    --recursive \
    $ROOT_REPO_DIR/runtime/chalicelib/www/ s3://pfun-cma-model-www/
sleep 0.1s

echo 'done'
