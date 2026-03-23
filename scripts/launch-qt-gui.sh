#!/usr/bin/env sh

# scripts/launch-qt-gui.sh

set -e

start_pfun_api_server() {
    # Start the pfun-cma-model server in the background
    echo "starting the pfun-cma-model API server..."
    pkill pfun-cma-model && sleep 1s
    nohup ./scripts/serve-dev.sh &
}

launch_qt_gui() {
    # Launch the Qt GUI
    echo "launching the Qt GUI..."
    pfun-qt-gui
}

echo -e "#################################"

# optionally: start_pfun_api_server
if [ "$1" = '-s' ];
then
     start_pfun_api_server
fi

echo -e ""
echo -e "#################################"
echo -e "#################################"
echo -e ""

# (always) launch the Qt GUI
launch_qt_gui

echo -e "#################################"
