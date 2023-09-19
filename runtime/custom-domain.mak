# Makefile for Chalice Custom Domain Setup

# Variables
DOMAIN_NAME = your_domain.com
STAGE_NAME = dev
APP_NAME = your_app_name
CERT_ARN = your_certificate_arn

# Create a new Chalice project
init:
	chalice new-project $(APP_NAME)

# Deploy the app
deploy:
	chalice deploy --stage $(STAGE_NAME)

# Create a custom domain
create_domain:
	aws apigateway create-domain-name \
		--domain-name $(DOMAIN_NAME) \
		--certificate-arn $(CERT_ARN)

# Map the custom domain to the API Gateway
map_domain:
	aws apigateway create-base-path-mapping \
		--domain-name $(DOMAIN_NAME) \
		--rest-api-id $(shell aws apigateway get-rest-apis --query "items[?name=='$(APP_NAME)-$(STAGE_NAME)'].id | [0]" --output text) \
		--stage $(STAGE_NAME)

# Delete the custom domain mapping
delete_mapping:
	aws apigateway delete-base-path-mapping \
		--domain-name $(DOMAIN_NAME) \
		--base-path '(none)'

# Delete the custom domain
delete_domain:
	aws apigateway delete-domain-name \
		--domain-name $(DOMAIN_NAME)

# Delete the Chalice app
delete_app:
	chalice delete --stage $(STAGE_NAME)

# Full setup: init -> deploy -> create_domain -> map_domain
full_setup: init deploy create_domain map_domain

# Full teardown: delete_mapping -> delete_domain -> delete_app
full_teardown: delete_mapping delete_domain delete_app
