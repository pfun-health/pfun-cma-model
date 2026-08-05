#!/usr/bin/env bash
#
# install.sh - Install the pfun-cma-model CLI globally
#
# This script:
#   1. Checks prerequisites (node, pnpm)
#   2. Installs dependencies via pnpm install
#   3. Builds all packages via pnpm build
#   4. Creates a wrapper script so 'pfun-cma-model' is available on PATH
#
# Usage:
#   ./install.sh              # Install the CLI
#   ./install.sh --uninstall  # Remove the installed wrapper
#

set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$SCRIPT_DIR"
BIN_NAME="pfun-cma-model"
CLI_ENTRY="$PROJECT_ROOT/packages/cli/dist/index.js"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { printf "${GREEN}==>${NC} %s\n" "$1"; }
warn()  { printf "${YELLOW}==>${NC} %s\n" "$1"; }
error() { printf "${RED}==>${NC} %s\n" "$1"; }

# --- Determine install directory ---
get_install_dir() {
    if [ -n "${PREFIX:-}" ]; then
        echo "${PREFIX}/bin"
    else
        echo "${HOME:?HOME not set}/.local/bin"
    fi
}

# --- Uninstall ---
if [ "${1:-}" = "--uninstall" ]; then
    INSTALL_DIR="$(get_install_dir)"
    WRAPPER="$INSTALL_DIR/$BIN_NAME"
    if [ -f "$WRAPPER" ]; then
        rm -f "$WRAPPER"
        rmdir "$INSTALL_DIR" 2>/dev/null || true
        info "Removed $BIN_NAME from $INSTALL_DIR"
    else
        warn "$BIN_NAME not found in $INSTALL_DIR (nothing to uninstall)"
    fi
    exit 0
fi

# --- Prerequisites ---
info "Checking prerequisites..."
if ! command -v node &>/dev/null; then
    error "Node.js is required but not found."
    error "Install from https://nodejs.org/ or use your package manager."
    exit 1
fi

node_version="$(node --version | sed 's/v//' | cut -d. -f1)"
if ! [[ "$node_version" =~ ^[0-9]+$ ]]; then
    error "Could not parse Node.js version from: $(node --version)"
    exit 1
fi
if [ "$node_version" -lt 18 ]; then
    error "Node.js >= 18 is required (found v$(node --version))."
    exit 1
fi

if ! command -v pnpm &>/dev/null; then
    error "pnpm is required but not found."
    error "Install via: npm install -g pnpm"
    exit 1
fi

# --- Install & Build ---
info "Installing project dependencies..."
pnpm install

info "Building packages..."
pnpm build

# Verify CLI entry exists
if [ ! -f "$CLI_ENTRY" ]; then
    error "Build output not found at $CLI_ENTRY."
    error "The build may have failed. Check for errors above."
    error "Try running: pnpm --filter cli build"
    exit 1
fi

# --- Install the wrapper script ---
INSTALL_DIR="$(get_install_dir)"
mkdir -p "$INSTALL_DIR"

WRAPPER="$INSTALL_DIR/$BIN_NAME"

# Create wrapper script that runs the CLI via node
# We embed the absolute project path so it works from any directory.
# The CLI's ES module imports (@pfun/core, @pfun/api) resolve through
# pnpm's workspace node_modules at packages/cli/node_modules/.
cat > "$WRAPPER" << WRAPPER_EOF
#!/usr/bin/env bash
# pfun-cma-model - Wrapper installed by install.sh
# Project: $PROJECT_ROOT
set -euo pipefail
exec node "$CLI_ENTRY" "\$@"
WRAPPER_EOF

chmod +x "$WRAPPER"

info "Installed $BIN_NAME to $INSTALL_DIR"

# Check if install dir is on PATH
path_has_dir() {
    case ":$PATH:" in
        *:"$1":*) return 0 ;;
        *)        return 1 ;;
    esac
}

if ! path_has_dir "$INSTALL_DIR"; then
    warn "$INSTALL_DIR is not in your PATH."
    warn "Add it to your shell configuration:"
    echo ""
    echo "    export PATH=\"$INSTALL_DIR:\$PATH\""
    echo ""
fi

info "Installation complete!"
info "Run 'pfun-cma-model --help' to get started."
