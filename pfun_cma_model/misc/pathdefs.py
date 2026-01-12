import logging

# initialize logger
logger = logging.getLogger("pfun_cma_model")
logger.setLevel(level=logging.INFO)
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import pfun_path_helper as pph  # type: ignore

pph.get_lib_path("pfun_cma_model")
from pfun_common.settings import get_settings
from pfun_common.utils import setup_logging

# Initialize logging based on settings
setup_logging(logger=logger, debug_mode=get_settings().debug)

__all__ = ["PFunDataPaths", "PFunAPIRoutes"]


@dataclass
class PFunDataPaths:
    """Paths for data files used in the pfun_cma_model package."""

    _pfun_data_dirpath: os.PathLike = Path(
        os.path.abspath(pph.get_lib_path("pfun_common")))
    _sample_data_fpath: os.PathLike = Path(
        os.path.join(_pfun_data_dirpath, "data/valid_data.csv")
    )
    _remote_data_fpath: str = (
        "https://github.com/pfun-health/pfun-data/releases/download/0.1.4/valid_data.csv"
    )

    def remove_sample_data(self) -> None:
        """Remove the sample data file if it exists."""
        if os.path.exists(self._sample_data_fpath):
            os.remove(self._sample_data_fpath)
            logger.debug(f"Sample data file {self._sample_data_fpath} removed.")
        else:
            logger.warning(
                f"(attempted to remove sample data) Sample data file {self._sample_data_fpath} does not exist. No action taken."
            )
        return

    def download_sample_data(self, overwrite: bool = False) -> None:
        """Download sample data from the remote file path."""
        if os.path.exists(self._sample_data_fpath) and not overwrite:
            logger.info(
                f"Sample data already exists at {self._sample_data_fpath}. Skipping download."
            )
            return
        with httpx.Client(follow_redirects=True, max_redirects=2) as client:
            response = client.get(self._remote_data_fpath)
            if response.status_code == 200:
                with open(self._sample_data_fpath, "wb") as f:
                    f.write(response.content)
                logger.info(f"Sample data downloaded to {self._sample_data_fpath}")
            else:
                raise Exception(
                    f"Failed to download sample data: {response.status_code}"
                )

    @property
    def sample_data_fpath(self) -> Path:
        return Path(self._sample_data_fpath)

    @property
    def pfun_data_dirpath(self) -> Path:
        return Path(self._pfun_data_dirpath)

    @property
    def remote_data_fpath(self) -> str:
        return self._remote_data_fpath

    def read_sample_data(self, fpath: Optional[os.PathLike] = None):
        """Read sample data from the specified file path."""
        if fpath is None:
            fpath = self.sample_data_fpath
        df = pd.read_csv(fpath)
        return df


@dataclass
class PFunAPIRoutes:
    FRONTEND_ROUTES = ("/run", "/run-at-time", "/params/schema", "/params/default")

    PUBLIC_ROUTES = (
        "/",
        "/model/run",
        "/model/fit",
        "/model/run-at-time",
        "/params/schema",
        "/params/default",
    )

    PRIVATE_ROUTES = ...
