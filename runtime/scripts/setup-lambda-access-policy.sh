#!/bin/bash
aws iam put-role-policy --role-name pfun-cma-model-dev --policy-document file://run_model_policy.json --policy-name pfun-cma-model-dev-api
