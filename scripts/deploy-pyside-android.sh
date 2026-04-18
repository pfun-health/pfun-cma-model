#!/usr/bin/env sh

set -e

SCRIPT_DIR=$(dirname "$0")

EXTRA_WHEELS_DIR=$(realpath "$SCRIPT_DIR/../packages/pfun_qt_gui/extra_wheels")
PYSIDE_SRC_DIR=$(realpath "$SCRIPT_DIR/../packages/pfun_qt_gui/src/pfun_qt_gui")

cd "$PYSIDE_SRC_DIR"

# Use Java 17 for Gradle 8.14.3 and Android SDK (Java 11 is too old for SDK,
# Java 26 not supported by Gradle).
if [ -d /usr/lib/jvm/java-17-openjdk ]; then
    export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
elif [ -d /usr/lib/jvm/java-11-openjdk ]; then
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
fi

# Skip JDK version check for Android SDK tools (they require Java 17+ but we use Java 11)
export SKIP_JDK_VERSION_CHECK=1

uv run \
    --python 3.11 \
    pyside6-android-deploy \
    --wheel-pyside "$EXTRA_WHEELS_DIR/pyside6-6.11.0-6.11.0-cp311-cp311-android_aarch64.whl" \
    --wheel-shiboken "$EXTRA_WHEELS_DIR/shiboken6-6.11.0-6.11.0-cp311-cp311-android_aarch64.whl" \
    --ndk-path ~/Android/Sdk/ndk/26.1.10909125 \
    --config-file "$PYSIDE_SRC_DIR/pysidedeploy.spec"
