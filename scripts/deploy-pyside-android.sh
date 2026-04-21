#!/usr/bin/env sh

set -e

export ROOT_DIR=$(realpath "$PWD")
echo "Root dir: $ROOT_DIR"
sleep 1s
export SCRIPT_DIR="${ROOT_DIR}/scripts"

export EXTRA_WHEELS_DIR=$(realpath "$ROOT_DIR/packages/pfun_qt_gui/extra_wheels")
export PYSIDE_SRC_DIR=$(realpath "$ROOT_DIR/packages/pfun_qt_gui/src/pfun_qt_gui")

cd "$PYSIDE_SRC_DIR"

# Skip JDK version check for Android SDK tools (they require Java 17+ but we use Java 11)
export SKIP_JDK_VERSION_CHECK=1

nohup sh -c 'yes | uv run \
    --python 3.11 \
    pyside6-android-deploy \
    --wheel-pyside "$EXTRA_WHEELS_DIR/pyside6-6.11.0-6.11.0-cp311-cp311-android_aarch64.whl" \
    --wheel-shiboken "$EXTRA_WHEELS_DIR/shiboken6-6.11.0-6.11.0-cp311-cp311-android_aarch64.whl" \
    --ndk-path ~/Android/Sdk/ndk/26.1.10909125 \
    --config-file "$PYSIDE_SRC_DIR/pysidedeploy.spec" \
    --keep-deployment-files
' > "$SCRIPT_DIR/../logs/pyside6-android-deploy.log" &

echo "log: $(realpath $SCRIPT_DIR/../logs/pyside6-android-deploy.log)"
