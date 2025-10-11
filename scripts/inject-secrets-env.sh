#!/usr/bin/env bash

# inject-secrets-env.sh

set -e

# inject secrets into the .env file
dcli inject \
	--in "./.env.template" \
	--out "./.env"
