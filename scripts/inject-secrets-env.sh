#!/usr/bin/env sh

# inject-secrets-env.sh

set -e

# try to find dcli
DCLI=$(which dcli)

if [[ -z $DCLI ]]; then
    echo "dashlane cli not installed (exiting!)"
    exit 1
fi

TEMPLATE_FN='./.env.template'
OUTPUT_FN='./.env'

echo "(from template: ${TEMPLATE_FN})"
echo "Injecting secrets into $OUTPUT_FN..."

# inject secrets into the .env file
$DCLI inject \
	--in "${TEMPLATE_FN}" \
	--out "${OUTPUT_FN}"
