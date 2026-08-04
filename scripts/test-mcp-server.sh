#!/usr/bin/env bash
#
# scripts/test-mcp-server.sh
#
# Smoke-test the MCP server: confirms the endpoint is reachable AND actually
# speaking MCP by performing a real JSON-RPC `initialize` handshake, not just
# a plain HTTP GET.
#
# URL precedence (first match wins):
#   1. first positional argument:  ./scripts/test-mcp-server.sh http://host:port/mcp
#   2. MCP_URL environment variable
#   3. default:                    http://localhost:8001/mcp

set -euo pipefail

readonly DEFAULT_MCP_URL="http://localhost:8001/mcp"
readonly MCP_TIMEOUT_SECONDS=10
readonly INITIALIZE_REQUEST='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test-mcp-server","version":"1.0.0"}}}'

# Global so the EXIT trap can reach it after main() returns.
response_file=""

cleanup() {
    if [[ -n "${response_file:-}" ]]; then
        rm -f "$response_file"
    fi
}

print_help() {
    cat <<'EOF'
Usage: test-mcp-server.sh [URL]

Smoke-test that the MCP server is reachable and speaking MCP.

Performs a JSON-RPC `initialize` handshake against the MCP endpoint and
verifies the response is a valid MCP response.

Arguments:
  URL   MCP endpoint to test (default: http://localhost:8001/mcp)

URL precedence (first match wins):
  1. Positional argument
  2. MCP_URL environment variable
  3. Default (http://localhost:8001/mcp)

Options:
  -h, --help   Show this help message and exit

Environment:
  MCP_URL      Overrides the default MCP endpoint URL
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

main() {
    local mcp_url="${1:-${MCP_URL:-$DEFAULT_MCP_URL}}"
    local http_status=""

    case "${1:-}" in
        -h | --help)
            print_help
            exit 0
            ;;
    esac

    response_file=$(mktemp) || fail "could not create a temporary file"
    trap cleanup EXIT

    printf 'Testing MCP server at %s ...\n' "$mcp_url"

    if ! http_status=$(curl -sS --max-time "$MCP_TIMEOUT_SECONDS" \
        -o "$response_file" \
        -w '%{http_code}' \
        -X POST "$mcp_url" \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' \
        -d "$INITIALIZE_REQUEST"); then
        fail "curl could not reach $mcp_url (is the server running there?)"
    fi

    if [[ "$http_status" != "200" ]]; then
        fail "expected HTTP 200 from $mcp_url but got HTTP $http_status"
    fi

    # The initialize reply is a JSON-RPC response; streamable HTTP may wrap it
    # in SSE framing (a leading "data: " line), so check substrings rather
    # than parsing JSON.
    if ! grep -Eq '"jsonrpc"' "$response_file" || ! grep -Eq '"serverInfo"|"result"' "$response_file"; then
        fail "response from $mcp_url is not a valid MCP initialize response"
    fi

    printf 'OK: MCP server at %s is reachable and answered the initialize handshake.\n' "$mcp_url"
    printf 'Response body:\n'
    cat "$response_file"
}

main "$@"
