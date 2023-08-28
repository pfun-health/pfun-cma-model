#!/bin/sh

export CC=gcc

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
echo -e "$(pkg-config --modversion minpack)"

# activate python3.10 venv
source /tmp/venv310/bin/activate

export MINPACK_SITE=/tmp/venv310/
PYTHON_VERSION=/tmp/venv310/bin/python

cd $HOME/Git/minpack || exit
fpm build
sleep 1s
meson setup --reconfigure _build -Dpython_version=$PYTHON_VERSION || exit
sleep 1s
meson install -C _build || exit
sleep 1s

cd $HOME/Git/minpack/python || exit
python -m pip install --upgrade cffi

meson setup --wipe --reconfigure _build -Dpython_version=$PYTHON_VERSION || exit
meson compile -C _build || exit
meson configure _build --prefix=$MINPACK_SITE || exit
meson install -C _build || exit

cd ~ && python -c $'import minpack\nprint(dir(minpack))'

cd - || exit
echo "...done install minpack python extension."
