#!/usr/bin/env bash

# inject-secrets-env.sh

set -e

DCLI="$(which dcli)"

if [[ -z $DCLI ]]; then
    echo -e "dashlane cli not installed (exiting!)"
    exit 1
done

TEMPLATE_FN='./.env.template'
OUTPUT_FN='./.env'

echo -e "(from template: ${TEMPLATE_FN})"
echo -e "Injecting secrets into $OUTPUT_FN..."

# inject secrets into the .env file
$DCLI inject \
	--in "${TEMPLATE_FN}" \
	--out "${OUTPUT_FN}"
