import subprocess
import os
import importlib
import logging
import sys


def get_pkg_config_libs_only_L(package_name):
    """
    Get the library path for a given package using pkg-config.
    Args:
        package_name (str): The name of the package to query.
    Returns:
        str: The library path for the package.
    """
    # Check if pkg-config is available
    try:
        # Call pkg-config and get the output
        output = subprocess.check_output(['pkg-config', '--libs-only-L', package_name], text=True)
        return output.strip()  # Remove any leading/trailing whitespace
    except subprocess.CalledProcessError as e:
        print(f"Error calling pkg-config: {e}")
        return None


def import_dynamic_module(module_name, lib_name=None):
    """
    Import a module that depends on a linked library by its name.
    Args:
        module_name (str): The name of the linked library to import.
        lib_name (str): The name of the library to load. If None, defaults to 'lib{module_name}.so'.
    Returns:
        module: The imported module.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Importing linked library: {module_name}")
    # Try to import the library
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        logger.warning(f"Failed to import {module_name}. Trying alternative method.")
    # Try to load the library using ctypes
    try:
        import ctypes
        libpath = get_pkg_config_libs_only_L(module_name)
        if not libpath:
            raise ImportError(f"Library path for {module_name} not found.")
        libpath = libpath.replace('-L', '').strip()
        libpath = libpath.replace(' ', os.pathsep)
        libpath = os.path.join(libpath, f'lib{module_name}.so')
        # Load the library using ctypes
        logger.debug(f"Loading library from path: {libpath}")
        # Check if the library path exists
        if not os.path.exists(libpath):
            raise ImportError(f"Library path {libpath} does not exist.")
        # Load the library using ctypes
        cdll = ctypes.cdll.LoadLibrary(libpath)
        module = importlib.import_module('minpack')
    except ImportError:
        # If the library is not found, raise an ImportError
        logger.warning(f"Failed to load {module_name} using ctypes.")
        raise ImportError(f"Failed to load {module_name}.")
