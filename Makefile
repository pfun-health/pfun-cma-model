ROOT_REPO_DIR = $(HOME)/Git/pfun-cma-model
CDK_OUT_DIR = $(ROOT_REPO_DIR)/infrastructure/cdk.out
STACK_NAME = PFunCMAModelChaliceApp
TEMPLATE_FILEPATH = $(CDK_OUT_DIR)/$(STACK_NAME).template.json
S3_BUCKET = pfun-cma-model-assets
AWS_REGION = us-east-1

.PHONY: get_Function_Id
get_Function_Id:
	cd $${CDK_OUT_DIR} || exit 1
	LAMBDA_TEST_FUNCTION_ID=$(shell sam list resources -t ${TEMPLATE_FILEPATH} --stack-name ${STACK_NAME} --output json | jq '.[] | select(.LogicalResourceId == "RunModel") | .PhysicalResourceId')
	cd $${ROOT_REPO_DIR} || exit 1

.PHONY: synthesize_local
synthesize_local:
	cd $(ROOT_REPO_DIR)/infrastructure && cdk synth --build --no-staging && cd $(ROOT_REPO_DIR)

.PHONY: validate
validate:
	sam validate -t $(TEMPLATE_FILEPATH)

.PHONY: synthesize_deploy
synthesize_deploy:
	cd $(ROOT_REPO_DIR)/infrastructure && cdk synth && cd $(ROOT_REPO_DIR)

.PHONY: build
build:
	sam build --manifest $(ROOT_REPO_DIR)/manifest.json -t $(TEMPLATE_FILEPATH)

.PHONY: package
package:
	sam package \
		--s3-bucket $(S3_BUCKET) \
		-t $(TEMPLATE_FILEPATH) \
		 --output-template-file $(ROOT_REPO_DIR)/packaged.yaml

.PHONY: deploy
deploy:
	sam deploy \
		--template-file $(ROOT_REPO_DIR)/packaged.yaml \
		--stack-name $(STACK_NAME) \
		--region $(AWS_REGION) \
		--capabilities CAPABILITY_IAM

.PHONY: test_local
test_local: get_Function_Id
	cd ${ROOT_REPO_DIR}/runtime && chalice local

test_local_sam: get_Function_Id
	sam local start-api -p 8000 -t $(TEMPLATE_FILEPATH)
	sam local invoke $(LAMBDA_TEST_FUNCTION_ID) -t $(TEMPLATE_FILEPATH)

.PHONY: test_deployment
test_deployment: get_Function_Id
	sam remote invoke $(LAMBDA_TEST_FUNCTION_ID) --stack-name $(STACK_NAME) --region $(AWS_REGION)

.PHONY: all
all: synthesize_local validate synthesize_deploy build package deploy test_deployment

.PHONY: clean
clean:
	set +e && rm -f $(ROOT_REPO_DIR)/packaged.yaml && set -e
	rm -rf $(CDK_OUT_DIR)/*
