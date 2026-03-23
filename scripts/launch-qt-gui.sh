#!/usr/bin/env sh

# scripts/launch-qt-gui.sh

set -e

# Start the pfun-cma-model server in the background
echo "starting the pfun-cma-model API server..."
nohup ./scripts/serve-prod.sh &

# Launc the Qt GUI
echo "launching the Qt GUI..."
sh -c 'cd packages/pfun_qt_gui; uv sync; python src/pfun_qt_gui/main.py'
