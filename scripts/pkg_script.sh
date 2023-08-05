#!/usr/bin/env bash

# package the dependencies, send to lambda layer

# fail on errors
set -e

echo "Packaging dependencies & updating lambda layer..."

cd ./lambda_layer/packages || exit

echo "installing to prefix..."

# install to prefix
pip install \
	--platform manylinux2014_x86_64 \
	--target=./python \
	--implementation cp \
	--python-version 3.10 \
	--only-binary=:all: \
	--upgrade \
	-r ../../requirements.txt

echo "...done installing to prefix."

echo "cleaning up dependencies..."

# cleanup
set +e
find . -name "*.dist-info" -exec rm -rf {} \;
find . -name "*.egg-info" -exec rm -rf {} \;
find . -name "*.pyc" -exec rm -rf {} \;
find . -name "__pycache__" -exec rm -rf {} \;
find . -name "tests" -exec rm -rf {} \;
find . -name "docs" -exec rm -rf {} \;
find . -name "doc" -exec rm -rf {} \;
echo "...done cleaning up dependencies."
set -e

echo "zipping..."

# zip
zip -rq python python

echo "...done zipping."

echo "uploading zip to s3..."

# upload to s3
aws --profile robbie --region us-east-1 \
	s3 cp python.zip s3://pfun-app-lambda/cma-model-deps.zip

echo "...done uploading zip to s3."

echo "publishing new lambda layer version..."

# publish lambda layer version
aws --profile robbie --region us-east-1 lambda publish-layer-version \
	--layer-name cma-model-deps \
	--description "dependencies for pfun cma model backend" \
	--content S3Bucket=pfun-app-lambda,S3Key=cma-model-deps.zip \
	--compatible-runtimes python3.10 python3.11 \
	--compatible-architectures "x86_64"

echo "...done publishing new dependency lambda layer version."

echo "cleaning up..."

# cleanup
rm -r ./python
rm python.zip

echo "...done cleaning up."

echo -e "\n...done updating dependency lambda layer.\n"

# return to root directory
cd - || exit
