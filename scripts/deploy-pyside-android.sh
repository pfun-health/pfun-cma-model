#!/usr/bin/env sh

set -e

EXTRA_WHEELS_DIR=$(realpath $SCRIPT_DIR/../packages/pfun_qt_gui/extra_wheels)
PYSIDE_SRC_DIR=$(realpath $SCRIPT_DIR/../packages/pfun_qt_gui/src/pfun_qt_gui)

sh -c "
cd $PYSIDE_SRC_DIR

uv run \
   --python 3.11 \
   pyside6-android-deploy \
   --wheel-pyside $EXTRA_WHEELS_DIR/pyside6-6.11.0-6.11.0-cp311-cp311-android_aarch64.whl \
   --wheel-shiboken $EXTRA_WHEELS_DIR/shiboken6-6.11.0-6.11.0-cp311-cp311-android_aarch64.whl \
   --ndk-path ~/Android/Sdk/ndk/26.1.10909125 \
   --config-file $PYSIDE_SRC_DIR/pysidedeploy.spec

cd -"
