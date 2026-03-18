#!/usr/bin/env bash

# generate-dev-certs.sh
# Script to generate self-signed development certificates
# Reference: https://www.compilenrun.com/docs/framework/fastapi/fastapi-deployment/fastapi-ssltls-setup/

export CERTS_DIR="./certs"

set -e

echo -e "Generating self-signed development certificates..."

# Create a directory for certificates
mkdir -p "${CERTS_DIR}"

generate_certs_openssl() {
	# Generate a private key and self-signed certificate
	openssl req -x509 \
		-newkey rsa:4096 \
		-keyout "${CERTS_DIR}/key.pem" \
		-out "${CERTS_DIR}/cert.pem" \
		-days 365 \
		-nodes
}

generate_certs_tailscale() {
	# Generate a private key and self-signed certificate using Tailscale's cert tool
	tailscale cert \
		--cert-file "${CERTS_DIR}/cert.pem" \
		--key-file "${CERTS_DIR}/key.pem" \
		"$(hostname).tail38611b.ts.net"
}

echo -e "Certificates generated in ${CERTS_DIR}/"
