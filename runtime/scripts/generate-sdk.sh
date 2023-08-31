#!/bin/bash

echo 'generating sdk...'

$HOME/Git/pfun-cma-model/runtime/scripts/sample-endpoint.sh sdk "-O" || exit
sleep 0.1s

cd $HOME/Git/pfun-cma-model/runtime &&
    chalice generate-models >./chalicelib/www/openapi.json || exit &&
    cd - || exit
sleep 0.1s

aws s3 cp \
    --recursive \
    $HOME/Git/pfun-cma-model/runtime/chalicelib/www/ s3://pfun-cma-model-www/
sleep 0.1s

echo 'done'
