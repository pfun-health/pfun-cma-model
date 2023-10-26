#!/usr/bin/env python
import os
import site
import subprocess
import sys


def main():
    #: Step 0: Install fortran dependencies
    print("Installing Fortran dependencies...")
    subprocess.run(
        ["sudo", "apt", "install", "-yy", "gfortran", "g++", "meson"], check=True
    )
    subprocess.run(["pip", "install", "--upgrade", "fpm", "ninja"], check=True)
    subprocess.run(["rm", "-rf", "minpack"], check=True)
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
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH"
"""
        bashpath = os.path.expanduser("~/.bashrc")
        if "# paths for minpack" not in open(bashpath, "r", encoding="utf-8").read():
            print("adding minpack paths to .bashrc...")
            with open(bashpath, "a", encoding="utf-8") as f:
                f.write(new_lines)
            print(f"...added paths:\n\t{new_lines}")

    print("Installing python module...")
    os.environ["LD_LIBRARY_PATH"] = os.path.join(prefix, "lib")
    if os.path.exists(os.path.join(os.environ['LD_LIBRARY_PATH'], "x86_64-linux-gnu")):
        os.environ['LD_LIBRARY_PATH'] = os.path.join(
            os.environ['LD_LIBRARY_PATH'], "x86_64-linux-gnu")
    os.environ["PKG_CONFIG_PATH"] = os.path.join(os.environ['LD_LIBRARY_PATH'], "pkgconfig")
    # subprocess.run(["python", "setup.py", "install"], check=True)

    os.chdir("..")
    subprocess.run(["pip", "install", "./dist/minpack-2.0.0-cp310-cp310-linux_x86_64.whl"])
    print("...finished installing python module.")

    print("...switched back to minpack directory.")

    # fail if minpack is not installed properly
    print("checking python module installation...")
    subprocess.run(["python", "-c", "from minpack import lmdif"], check=True)
    print("...success.")

    # Step 5: Clean up
    os.chdir("..")
    print("...switched back to repo root.")
    print("cleaning up...")
    subprocess.run(["rm", "-rf", "minpack"], check=True)
    print("...done.")


if __name__ == "__main__":
    main()
