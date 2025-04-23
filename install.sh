#!/usr/bin/env bash

# install.sh

# exit on error
set -e

# get the python version
PYTHON_VERSION=$(which python3)
echo -e "(detected) Python version: $PYTHON_VERSION"

# define the build directory
BUILDDIR=_build

# Setup the meson build system
# clean up the build directory
if [ -d $BUILDDIR ]; then
    echo -e "Cleaning up the build directory..."
    rm -rf $BUILDDIR
fi
# re-create the build directory
mkdir -p $BUILDDIR
echo -e "Setting up the meson build system..."
meson setup --reconfigure $BUILDDIR -Dpython_version=$PYTHON_VERSION

# Compile the project
# This will create a build directory and compile the project
echo -e "Compiling the project..."
meson compile -C $BUILDDIR

# Install the project
# This will configure & install the project to the specified prefix (e.g., --prefix=/usr/local)
echo -e "Configuring the project for installation..."
PYTHON_PREFIX=$(python3 -c "import sys; print(sys.prefix)")
echo -e "(detected) Python prefix: $PYTHON_PREFIX"
meson configure $BUILDDIR --prefix=$PYTHON_PREFIX
# Install the project
echo -e "Installing the project..."
meson install -C $BUILDDIR