#!/usr/bin/env python
import platform
import os
import site
import subprocess
import sys
import importlib


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
            ["sudo", "apt", "install", "-yy", "gfortran", "g++", "meson"], check=True
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
    subprocess.run(["pip", "install", "--upgrade", "fpm", "ninja"], check=True)
    try:
        subprocess.run(["rm", "-rf", "minpack"], check=False)
    except:
        pass
    # Step 1: Clone the repository
    subprocess.run(
        ["git", "clone", "https://github.com/rocapp/minpack.git"], check=True
    )

    # Change directory to the cloned repository
    os.chdir("minpack")
    print("...switched to minpack directory.")

    # Step 2, 3: build the Fortran library & Python module
    print("building Fortran library...")
    python_version = os.path.join(sys.prefix, "bin", "python")
    prefix = site.getuserbase()
    subprocess.run(
        ["meson", "setup", "_build", "-Dpython=true", f"-Dprefix={prefix}"], check=True
    )
    subprocess.run(
        [
            "meson",
            "--reconfigure",
            f"-Dpython_version={python_version}",
            "-Dpython=true",
            f"--prefix={prefix}",
            "_build",
        ],
        check=True,
        shell=False,
    )
    subprocess.run(["meson", "compile", "-C", "_build"], check=True)

    # Step 4: Install the Python module
    print("building Python module dependencies...")
    subprocess.run(["meson", "install", "-C", "_build"], check=True)

    os.chdir("python")
    print("...switched to python directory.")

    # set the paths (if not already set)
    try:
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
    os.environ["LD_LIBRARY_PATH"] = os.path.join(prefix, "lib")
    if os.path.exists(os.path.join(os.environ['LD_LIBRARY_PATH'], "x86_64-linux-gnu")):
        os.environ['LD_LIBRARY_PATH'] = os.path.join(
            os.environ['LD_LIBRARY_PATH'], "x86_64-linux-gnu")
    os.environ["PKG_CONFIG_PATH"] = os.path.join(
        os.environ['LD_LIBRARY_PATH'], "pkgconfig")
    subprocess.run(["poetry", "run", "python", "setup.py", "install"], check=True, env=os.environ)
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
