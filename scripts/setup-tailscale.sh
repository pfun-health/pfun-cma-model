#!/usr/bin/env sh

# scripts/setup-tailscale.sh
# setup tailscale funnel, certificates for HTTPS

set -e

setup_funnel() {
    tailscale funnel --bg 443 \
    https://localhost:8443
}

export_certs() {
    tailscale cert # TODO: finish this...
}

install_ts_nginx_auth() {
    # From here you can configure your
    # applications to use the contents of
    # 'X-Webauth-User' or the other headers
    # to use that for authentication logic.
    sudo sh -c \
    "apt-get update && apt-get install tailscale-nginx-auth && sleep 1s; systemctl enable --now tailscale.nginx-auth.socket"
}