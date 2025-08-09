# tests/test_base.py
# This file sets up the test environment for the pfun-cma-model package.
import sys
from pathlib import Path


def setup_test_environment():
    """Sets up the test environment by adding the root and module paths to sys.path.
    This allows the tests to import modules from the pfun-cma-model package.
    """
    # Get the root path of the project (two levels up from this file)
    # and the module path (one level up from this file).
    root_path = str(Path(__file__).parents[2])
    mod_path = str(Path(__file__).parents[1])
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    if mod_path not in sys.path:
        sys.path.insert(0, mod_path)
    # Import the pfun_path_helper module to ensure it is available for use.
    import pfun_path_helper as pph  # type: ignore
    return root_path, mod_path, pph


root_path, mod_path, pph = setup_test_environment()
