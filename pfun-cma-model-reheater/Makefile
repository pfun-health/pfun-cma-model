# Makefile for Chalice App Deployment

# Variables
STAGE ?= dev
APP_NAME = pfun-cma-model-reheater

# Phony targets ensure that Make doesn't do something with a file named the same as a command
.PHONY: deploy package delete info clean local

# Deploy the Chalice app to the specified stage
deploy:
	chalice deploy --stage $(STAGE)

# Generate a deployment package for the Chalice app
package:
	chalice package --stage $(STAGE) ./package

# Delete the deployed app resources
delete:
	chalice delete --stage $(STAGE)

# Display information about the deployed app
info:
	chalice url --stage $(STAGE)

# Clean any generated artifacts
clean:
	rm -rf ./package
	rm -rf .chalice/deployments

# Run the app locally
local:
	chalice local

# Invoke a specific Lambda function (replace function_name)
invoke:
	chalice invoke -n function_name
