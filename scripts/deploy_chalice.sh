#!/usr/bin/env bash

set -e

echo -e "now packaging the chalice application...\n"

# package the chalice application -> ./packaged/
chalice package packaged || exit

echo -e "   ...done packaging the chalice application.\n"

# go to packaged directory
cd packaged || exit

echo "readying for deployment..."

# package the chalice application (-> ./sam-packaged.json)
aws --profile robbie --region us-east-1 \
	cloudformation package --template-file ./sam.json \
	--s3-bucket pfun-app-lambda \
	--output-template-file sam-packaged.yaml

echo -e "...done readying for deployment.\n"

echo "deploying the chalice application..."

# deploy the chalice application
aws --profile robbie --region us-east-1 \
	cloudformation deploy \
	--template-file /home/$USER/Git/pfun-cma-model/packaged/sam-packaged.yaml \
	--stack-name pfun-app-lambda \
	--capabilities CAPABILITY_IAM

echo -e "...done deploying the chalice application.\n"

echo -e "\n...done."

cd - || exit
