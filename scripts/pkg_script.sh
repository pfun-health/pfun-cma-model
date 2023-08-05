#!/usr/bin/env bash

# package the dependencies, send to lambda layer

# fail on errors
set -e

cd ./lambda_layer/packages || exit

# install to prefix
pip install -r ../requirements.txt -t ./python

# cleanup
find . -name "*.dist-info" -exec rm -rf {} \;
find . -name "*.egg-info" -exec rm -rf {} \;
find . -name "*.pyc" -exec rm -rf {} \;
find . -name "__pycache__" -exec rm -rf {} \;
find . -name "tests" -exec rm -rf {} \;
find . -name "docs" -exec rm -rf {} \;
find . -name "doc" -exec rm -rf {} \;

# zip
zip -r python python

# upload to s3
aws s3 cp python.zip s3://pfun-app-lambda/cma-model-deps.zip

# publish lambda layer version
aws --region us-east-1 lambda publish-layer-version \
	--layer-name cma-model-deps \
	--description "dependencies for pfun cma model backend" \
	--content S3Bucket=pfun-app-lambda,S3Key=cma-model-deps.zip \
	--compatible-runtimes python3.10 python3.11 \
	--compatible-architectures "x86_64"

# cleanup
rm -r ./python
rm python.zip

# return to previous directory
cd - || exit
