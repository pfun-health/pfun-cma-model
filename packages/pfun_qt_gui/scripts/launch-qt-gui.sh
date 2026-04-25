#!/usr/bin/env sh

# scripts/launch-qt-gui.sh

set -e

SCRIPT_DIR=$(realpath "$(dirname "$0")")
GUI_DIR=$(realpath "$SCRIPT_DIR/..")

start_pfun_api_server() {
    # Start the pfun-cma-model server in the background
    echo "starting the pfun-cma-model API server..."
    pkill pfun-cma-model && sleep 1s
    nohup "$SCRIPT_DIR/../../scripts/serve-dev.sh" &
}

launch_qt_gui() {
    # Launch the Qt GUI
    echo "Sync the GUI dependencies..."
    cd "$GUI_DIR" && "$SCRIPT_DIR/uv-full-sync.sh" -gui || echo "Sync failed, but continuing anyway..."
    sleep 1s
    echo "launching the Qt GUI..."
    cd "$GUI_DIR" && uv run python -m pfun_qt_gui.main
}

echo -e "#################################"

# optionally: start_pfun_api_server
if [ "$1" != '-N' ]; then # -N means No server
    start_pfun_api_server
    # wait for the server to be available...
    echo "waiting for the server to be available (just a few seconds)..."
    sleep 3s
fi

echo -e ""
echo -e "#################################"
echo -e "#################################"
echo -e ""

# (always) launch the Qt GUI
launch_qt_gui

echo -e "#################################"
