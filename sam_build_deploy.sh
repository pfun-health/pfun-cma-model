#!/bin/bash

export ROOT_REPO_DIR=$HOME/Git/pfun-cma-model
export CDK_OUT_DIR=$ROOT_REPO_DIR/infrastructure/cdk.out
export STACK_NAME=PFunCMAModelChaliceApp
export TEMPLATE_FILEPATH=$CDK_OUT_DIR/$STACK_NAME.template.json

get_Function_Id() {
    cd $CDK_OUT_DIR || exit 1
    JSON_OUTPUT=$(sam list resources --stack-name $STACK_NAME --output json) ||
        exit 1
    LAMBDA_TEST_FUNCTION_ID=$(
        echo $JSON_OUTPUT |
            jq '.[] |
          select(.LogicalResourceId == "RunModel") |
           .PhysicalResourceId'
    ) || exit 1
    export LAMBDA_TEST_FUNCTION_ID=${LAMBDA_TEST_FUNCTION_ID//\"/} ||
        exit 1
    cd - || exit 1
}

synthesize_local() {
    # synthesize (local)
    cd $ROOT_REPO_DIR/infrastructure || exit 1
    cdk synth --no-staging || exit 1
    cd - || exit 1
}

synthesize_deploy() {
    # synthesize (deploy)
    cd $ROOT_REPO_DIR/infrastructure || exit 1
    cdk synth --no-staging || exit 1
    cd - || exit 1
}

# initially synthesize for local testing
synthesize_local

# validate
sam validate -t $TEMPLATE_FILEPATH ||
    exit 1

# get function ID
get_Function_Id

# test local
sam local start-api -p 8000 -t $TEMPLATE_FILEPATH ||
    exit 1
sam local invoke $LAMBDA_TEST_FUNCTION_ID -t $TEMPLATE_FILEPATH ||
    exit 1

# if successful, re-synthesize for deployment
synthesize_deploy

# build
sam build --manifest $ROOT_REPO_DIR/manifest.json -t $TEMPLATE_FILEPATH ||
    exit 1

# package
sam package --s3-bucket s3://pfun-cma-model-assets \
    --output-template-file $ROOT_REPO_DIR/packaged.yaml ||
    exit 1

# deploy
sam deploy \
    --template-file /home/robertc/Git/pfun-cma-model/infrastructure/cdk.out/packaged.yaml \
    --stack-name $STACK_NAME \
    --region us-east-1 \
    --capabilities CAPABILITY_IAM ||
    exit 1

# test deployment
sam remote invoke \
    $LAMBDA_TEST_FUNCTION_ID \
    --stack-name $STACK_NAME \
    --region us-east-1
