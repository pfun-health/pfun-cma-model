#!/bin/bash

aws apigateway test-invoke-method --http-method POST --multi-value-headers $(./scripts/get-auth-headers.sh) ^Crest-api-id 2386q9818c --resourc
e-id 79ox19