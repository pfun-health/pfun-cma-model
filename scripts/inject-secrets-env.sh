#!/usr/bin/env bash

# inject-secrets-env.sh

set -e

DCLI='/home/robbiec/standalone-apps/dcli-linux-x64'

TEMPLATE_FN='./.env.template'
OUTPUT_FN='./.env'

echo -e "(from template: ${TEMPLATE_FN})"
echo -e "Injecting secrets into $OUTPUT_FN..."

# inject secrets into the .env file
$DCLI inject \
	--in "${TEMPLATE_FN}" \
	--out "${OUTPUT_FN}"
