#!/usr/bin/env sh

# packages/pfun_qt_gui/scripts/uv-full-sync-qt-gui.sh
# perform full uv sync (include Qt6 + other gui-related deps)

set -e

export GUI_DIR="$(realpath $(dirname $0)/..)"
echo "\nGUI_DIR: $GUI_DIR\n"

full_uv_sync_qt_gui() {
    # Perform a full uv sync for the qt-gui package.
    cd "$GUI_DIR"
    uv sync \
        --reinstall \
        --all-extras \
        --link-mode copy
}

full_uv_sync_qt_gui
