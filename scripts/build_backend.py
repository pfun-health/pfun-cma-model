import os
import subprocess
from setuptools.build_meta import build_sdist, build_wheel

def build_minpack():
    """Build and install libminpack."""
    repo_url = "https://github.com/fortran-lang/minpack.git"
    branch = "main"
    build_dir = os.path.join(os.getcwd(), "build", "minpack")
    install_dir = os.path.join(os.getcwd(), "build", "minpack_install")

    # Clone the repository if it doesn't exist
    if not os.path.exists(build_dir):
        subprocess.check_call(["git", "clone", "--branch", branch, repo_url, build_dir])

    # Build and install libminpack
    os.makedirs(install_dir, exist_ok=True)
    subprocess.check_call(["meson", "setup", "build", "--prefix", install_dir], cwd=build_dir)
    subprocess.check_call(["meson", "compile", "-C", "build"], cwd=build_dir)
    subprocess.check_call(["meson", "install", "-C", "build"], cwd=build_dir)

    # Update environment variables for runtime linking
    lib_path = os.path.join(install_dir, "lib")
    os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    print(f"libminpack installed to {install_dir}")

# Hook into the build process
def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    build_minpack()
    return build_wheel.prepare_metadata_for_build_wheel(metadata_directory, config_settings)

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    build_minpack()
    return build_wheel(wheel_directory, config_settings, metadata_directory)

def build_sdist(sdist_directory, config_settings=None):
    build_minpack()
    return build_sdist(sdist_directory, config_settings)