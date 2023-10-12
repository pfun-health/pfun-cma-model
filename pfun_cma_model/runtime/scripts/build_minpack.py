#!/usr/bin/env python
import subprocess
import os
import site
import sys


def main():
    subprocess.run(['rm', '-rf', 'minpack'], check=True)
    # Step 1: Clone the repository
    subprocess.run(['git', 'clone', 'https://github.com/rocapp/minpack.git'], check=True)

    # Change directory to the cloned repository
    os.chdir('minpack')

    # Step 2, 3: build the Fortran library & Python module
    python_version = os.path.join(sys.prefix, 'bin', 'python')
    prefix = site.getuserbase()
    subprocess.run(['meson', 'setup', '_build',
                    '-Dpython=true',
                    f'-Dprefix={prefix}'
                    ], check=True)
    subprocess.run(['meson', '--reconfigure', f'-Dpython_version={python_version}', '-Dpython=true', f'--prefix={prefix}', '_build'], check=True, shell=False)
    subprocess.run(['meson', 'compile', '-C', '_build'], check=True)
    
    # Step 4: Install the Python module
    subprocess.run(['meson', 'install', '-C', '_build'], check=True)
    os.chdir('python')
    subprocess.run(['python', 'setup.py', 'install'], check=True)
    os.chdir('..')

    # set the paths (if not already set)
    try:
        from minpack import lmdif
    except ImportError:
        new_lines = '''
# paths for minpack
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH"
'''
        bashpath = os.path.expanduser('~/.bashrc')
        if '# paths for minpack' not in open(bashpath, 'r', encoding='utf-8').read():
            print('adding minpack paths to .bashrc...')
            with open(bashpath, 'a', encoding='utf-8') as f:
                f.write(new_lines)

    # fail if minpack is not installed properly
    os.environ['LD_LIBRARY_PATH'] = os.path.join(prefix, 'lib')
    os.environ['PKG_CONFIG_PATH'] = os.path.join(prefix, 'lib', 'pkgconfig')
    subprocess.run(['python', '-c', 'from minpack import lmdif'], check=True)

    # Step 5: Clean up
    os.chdir('..')
    subprocess.run(['rm', '-rf', 'minpack'], check=True)


if __name__ == "__main__":
    main()
