import logging

# get the version via python standard library
import importlib.metadata
from .data import format_data, read_sample_data
from .engine.cma import CMASleepWakeModel
from .engine.cma_model_params import CMAModelParams
from .engine.cma_plot import CMAPlotConfig
from .engine.fit import fit_model
from .misc.pathdefs import PFunDataPaths

__all__ = [
    "PFunDataPaths",
    "CMAModelParams",
    "CMASleepWakeModel",
    "CMAPlotConfig",
    "fit_model",
    "read_sample_data",
    "format_data",
]

# top-level convenience imports


def get_version():
    """Get the version of the pfun-cma-model package."""
    version_ = importlib.metadata.version("pfun-cma-model")
    logging.debug("pfun-cma-model version: %s", version_)
    return version_


try:
    __version__ = get_version()
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
    logging.warning("pfun-cma-model package version not found. Using default version %s.", __version__)
