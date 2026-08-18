#!/usr/bin/env sh

# inject-secrets-env.sh

set -e

# try to find dcli
export DCLI="$(which dcli)"

if [ -z "$DCLI" ] || [ "$DCLI" == "dcli not found" ]; then
    echo -e "dashlane cli not installed (exiting!)"
    exit 1
fi

export TEMPLATE_FN="${PWD}/.env.template"
export OUTPUT_FN="${PWD}/.env"

echo "(from template: ${TEMPLATE_FN})"
echo "Injecting secrets into $OUTPUT_FN..."

# inject secrets into the .env file
$DCLI inject \
	--in "${TEMPLATE_FN}" \
	--out "${OUTPUT_FN}"
