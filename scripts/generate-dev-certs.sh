#!/usr/bin/env bash

# generate-dev-certs.sh
# Script to generate self-signed development certificates
# Reference: https://www.compilenrun.com/docs/framework/fastapi/fastapi-deployment/fastapi-ssltls-setup/

export CERTS_DIR="./certs"

set -e

detect_menu_tool() {
	if [[ "$(uname -a)" == *"NixOS"* ]]; then
		# utilize nix run
		echo "nix run nixpkgs#"
	fi
	if command -v whiptail &>/dev/null; then
		echo "whiptail"
	elif command -v dialog &>/dev/null; then
		echo "dialog"
	elif command -v select &>/dev/null; then
		echo "select"
	else
		echo "none"
	fi
}

show_menu() {
	local tool="$1"
	local choice
	local -a openssl_options
	local -a tailscale_options

	tailscale_options=("1" "Tailscale (Letsencrypt, served internal on the tailnet)")
	openssl_options=("2" "OpenSSL (self-signed, local)")

	if [[ "$tool" == "whiptail" ]]; then
		choice=$(whiptail --title "Generate Dev Certificates" \
			--menu "Choose certificate generation method:" 12 60 2 \
			"${tailscale_options[@]}" \
			"${openssl_options[@]}" 3>&1 1>&2)
	elif [[ "$tool" == "dialog" ]]; then
		choice=$(dialog --title "Generate Dev Certificates" \
			--menu "Choose certificate generation method:" 12 60 2 \
			"${tailscale_options[@]}" \
			"${openssl_options[@]}" 3>&1 1>&2)
	elif [[ "$tool" == "select" ]]; then
		echo "Certificate generation methods:" >&2
		echo "${tailscale_options[0]}) ${tailscale_options[1]}" >&2
		echo "${openssl_options[0]}) ${openssl_options[1]}" >&2
		echo ""
		read -p "Enter choice [1]: " choice
		choice="${choice:-1}"
	elif [[ "$tool" == "none" ]]; then
		echo "No menu tool detected. Falling back to command-line input." >&2
		echo "Certificate generation methods:" >&2
		echo "${tailscale_options[0]}) ${tailscale_options[1]}" >&2
		echo "${openssl_options[0]}) ${openssl_options[1]}" >&2
		echo "" >&2
		read -p "Enter choice [1]: " choice
		choice="${choice:-1}"
	else
		echo "Certificate generation methods:" >&2
		echo "${tailscale_options[0]}) ${tailscale_options[1]}" >&2
		echo "${openssl_options[0]}) ${openssl_options[1]}" >&2
		echo "" >&2
		read -p "Enter choice [1]: " choice
		choice="${choice:-1}"
	fi

	echo "$choice"
}

generate_certs_openssl() {
	echo -e "Generating self-signed certificate with OpenSSL..."

	openssl req -x509 \
		-newkey rsa:4096 \
		-keyout "${CERTS_DIR}/key.pem" \
		-out "${CERTS_DIR}/cert.pem" \
		-days 365 \
		-nodes \
		-subj "/CN=localhost/O=Development"

	echo -e "Done."
}

generate_certs_tailscale() {
	if ! command -v tailscale &>/dev/null; then
		echo "Error: tailscale command not found. Install Tailscale first." >&2
		return 1
	fi

	local domain_arg
	domain_arg="$(hostname).tail38611b.ts.net"

	echo -e "Generating certificate with Tailscale..."
	echo "  Host: '$domain_arg'"
	echo "  Note: Ensure you are logged into Tailscale."

	tailscale cert \
		--cert-file "${CERTS_DIR}/${domain_arg}.crt" \
		--key-file "${CERTS_DIR}/${domain_arg}.key" \
		"${domain_arg}"

	echo -e "Done."
}

main() {
	echo -e "Generating self-signed development certificates..."

	mkdir -p "${CERTS_DIR}"

	local menu_tool
	menu_tool=$(detect_menu_tool)

	local choice
	choice=$(show_menu "$menu_tool")

	case "$choice" in
	"1")
		generate_certs_tailscale
		;;
	"2")
		generate_certs_openssl
		;;
	*)
		echo "Invalid choice: $choice" >&2
		exit 1
		;;
	esac

	echo -e "Certificates generated in ${CERTS_DIR}/"
}

main "$@"
