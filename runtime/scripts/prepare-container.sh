#!/bin/bash

# prepare-container.sh

# prepare the dev container for pfun-cma-model
set -e

./scripts/create-venv.sh || exit 1

echo -e '...successfully created venv!'

conda install -r requirements-dev.txt -r requirements.txt

echo -e '...successfully installed dependencies!'

echo -e 'setup aws_completer...'

# trunk-ignore(shellcheck/SC2016)
echo 'export PATH="$PATH:/usr/local/bin"' >>"/home/${USER}/.bashrc"
# trunk-ignore(shellcheck/SC1090)
# trunk-ignore(shellcheck/SC2086)
source /home/${USER}/.bashrc
complete -C '/usr/local/bin/aws_completer' aws
echo 'complete -C /usr/local/bin/aws_completer aws' >>"/home/${USER}/.bashrc"
echo -e '...finished setting up aws_completer.'

echo -e '...done.'

exit 0
