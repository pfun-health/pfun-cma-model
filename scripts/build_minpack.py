#!/usr/bin/env python
import platform
import os
import site
import subprocess
import sys
import importlib


def get_app_root():
    """
    Returns the root directory of the application.

    This function checks the current working directory and returns the root directory
    of the application based on the presence of a specific file.

    Returns:
        str: The root directory of the application.
    """
    current_dir = os.getcwd()
    while True:
        if os.path.exists(os.path.join(current_dir, "install.sh")):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("'install.sh' not found in any parent directory.")
        current_dir = parent_dir


def install_arch_specific():
    """
    Installs architecture-specific packages based on the current platform.

    This function determines the package manager available on the current platform
    and installs the necessary packages accordingly. It supports Linux-based systems
    with either `apt` or `pacman` package managers.

    Parameters:
    None

    Returns:
    None
    """
    package_manager = None

    if platform.system() == "Linux":
        if subprocess.run(["which", "apt"], capture_output=True, text=True).returncode == 0:
            package_manager = "apt"
        elif subprocess.run(["which", "pacman"], capture_output=True, text=True).returncode == 0:
            package_manager = "pacman"

    if package_manager == "apt":
        subprocess.run(
            ["sudo", "apt", "install", "-yyq", "gfortran", "g++", "meson"], check=True
        )
    elif package_manager == "pacman":
        subprocess.run(
            ["sudo", "pacman", "-S", "--noconfirm", "gcc-fortran", "gcc", "meson"], check=True
        )
    else:
        print("Unsupported package manager.")


def main():
    #: Step 0: Install fortran dependencies
    print("Installing Fortran dependencies...")
    # install_arch_specific()  # commented out to avoid errors when building in docker
    subprocess.run(["python3", "-m", "pip", "install", "--upgrade", "fpm", "ninja", "meson"], check=True)
    print("...success.")

    print("getting app root directory...")
    # get the root directory of the application
    app_rootdir = get_app_root()
    print(f"...app root directory: {app_rootdir}")
    os.chdir(app_rootdir)
    print("...switched to app root directory.")
    # Step 1: Check if the minpack repository already exists
    if os.path.exists(os.path.join(app_rootdir, "minpack")):
        print("minpack directory already exists. No worries.")
        pass
    else:
        print("...no existing minpack directory found.")
        # Step 1: Clone the minpack repository (as a submodule)
        print("cloning minpack repository...")
        output = subprocess.run(
            ["git", "pull", "--recurse-submodules"], capture_output=True
        )
        if output.returncode != 0:
            print("...failed to clone minpack repository.")
            print(output.stderr.decode())
            raise SystemExit(1)
        print("...success.")

    # Step 2, 3: build the minpack Fortran library & Python module
    print("building minpack Fortran library & Python module...")
    print("...running 'MINPACK_ROOT/scripts/install_minpack.sh'...")
    # run the install script using Popen and stream stdout and stderr
    try:
        with subprocess.Popen(
            ["./scripts/install_minpack.sh"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.join(app_rootdir, "minpack"),  # ! change to the minpack directory
            env=os.environ,
            text=True
        ) as process:
            for line in process.stdout:
                print(line, end="")  # stream stdout
            stderr_output = process.communicate()[1]  # capture stderr after stdout finishes

            if process.returncode != 0:
                print("...failed to build minpack Fortran library & Python module.")
                print(stderr_output)
                raise subprocess.CalledProcessError(process.returncode, process.args, output=None, stderr=stderr_output)
            print("...success.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure the './scripts/install_minpack.sh' script exists and is executable.")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with return code {e.returncode}.")
        print(f"Command: {e.cmd}")
        print(f"Stderr: {e.stderr}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    # set the paths (if not already set)
    try:
        print("Testing if the minpack Python module is installed...")
        from minpack import lmdif
    except ImportError:
        new_lines = """
# paths for minpack
export LD_LIBRARY_PATH="$HOME/.local/lib:$HOME/.local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$HOME/.local/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH"
"""
        bashpath = os.path.expanduser("~/.bashrc")
        if "# paths for minpack" not in open(bashpath, "r", encoding="utf-8").read():
            print("adding minpack paths to .bashrc...")
            with open(bashpath, "a", encoding="utf-8") as f:
                f.write(new_lines)
            print(f"...added paths:\n\t{new_lines}")

    print("\n\nInstalling python module...")
    prefix = site.getuserbase()  # get the user base directory (e.g., ~/.local)
    os.environ["LD_LIBRARY_PATH"] = os.path.join(prefix, "lib")
    if os.path.exists(os.path.join(os.environ['LD_LIBRARY_PATH'], "x86_64-linux-gnu")):
        os.environ['LD_LIBRARY_PATH'] = os.path.join(
            os.environ['LD_LIBRARY_PATH'], "x86_64-linux-gnu")
    os.environ["PKG_CONFIG_PATH"] = os.path.join(
        os.environ['LD_LIBRARY_PATH'], "pkgconfig")
    subprocess.run(["python", "setup.py", "install"], check=True, env=os.environ)
    print("...success.")

    os.chdir("..")
    print("\n...switched back to minpack directory.")

    # fail if minpack is not installed properly
    print("\n\nchecking python module installation...")
    try:
        subprocess.run(["python", "-c", "from minpack import lmdif"], check=True, env=os.environ)
    except subprocess.CalledProcessError:
        minpack = importlib.import_module("minpack")
        print("...success.")
    else:
        print("...success.")
        return

if __name__ == "__main__":
    main()
