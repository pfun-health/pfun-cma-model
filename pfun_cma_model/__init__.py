import logging
from .misc.pathdefs import *
from .engine.cma_model_params import *
from .engine.cma import *
from .engine.fit import *
from .engine.cma_plot import *
from .data import *

__all__ = [
    "PFunDataPaths",
    "CMAModelParams",
    "CMASleepWakeModel",
    "CMAPlotConfig",
    "fit_model",
    "read_sample_data",
    "format_data"
]

# top-level convenience imports

# get the version via python standard library
import importlib.metadata
def get_version():
    """Get the version of the pfun-cma-model package."""
    version_ = importlib.metadata.version("pfun-cma-model")
    logging.debug(f"pfun-cma-model version: {version_}")
    return version_

try:
    __version__ = get_version()
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
    logging.warning(f"pfun-cma-model package version not found. Using default version {__version__}.")
