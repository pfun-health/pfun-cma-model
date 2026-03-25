import logging

# initialize logger
logger = logging.getLogger("pfun_cma_model.misc.pathdefs")
logger.setLevel(level=logging.INFO)
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import pfun_path_helper as pph
from pfun_common.settings import get_settings
from pfun_common.utils import setup_logging
from hashlib import sha256

# Initialize logging based on settings
setup_logging(logger=logger, debug_mode=get_settings().debug)

__all__ = [
    "PFunDataPaths",
]


@dataclass
class PFunDataPath:
    """Represents a file path for a data resource, along with its SHA256 hash for integrity verification (for the file)."""

    raw_path: os.PathLike | str
    #: The raw file path as provided. This can be a string or Path-like object.

    sha256_digest: Optional[str] = None
    #: The SHA256 hash of the file at the given path. This is calculated in the __post_init__ method after the object
    #   is initialized. It can be used for integrity verification of the file at the path. It is optional because it
    #   may not be known at initialization time, but will be calculated and set after the object is created.

    def __post_init__(self):
        try:
            _calculated_sha256_digest = self.calc_sha256(self.raw_path)
        except FileNotFoundError:
            logger.debug("(failed to calculate SHA256) File not found: %s", self.raw_path)
        else:
            # If sha256_digest was provided at initialization, it must match the calculated hash.
            #   Otherwise, if the sha256_digest was not provided, it will be set to the calculated hash.
            assert self.sha256_digest is None or self.sha256_digest == _calculated_sha256_digest, (
                f"SHA256 hash mismatch for file at {self.raw_path}. "
                f"Expected: {self.sha256_digest}, Calculated: {_calculated_sha256_digest}"
            )
            self.sha256_digest = _calculated_sha256_digest

    @classmethod
    def calc_sha256(cls, raw_path: os.PathLike | str) -> str:
        """Calculate and return the SHA256 hash of the file at the given path."""
        raw_path = str(raw_path)
        if not os.path.isfile(raw_path):
            raise FileNotFoundError(f"(failed to calculate SHA256) File not found: {raw_path}")
        sha256_hash = sha256()
        with open(raw_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @property
    def path(self) -> Path:
        """Return the Path object for the given raw path."""
        return Path(self.raw_path)


@dataclass
class PFunDataPaths:
    """Paths for data files used in the pfun_cma_model package."""

    _pfun_root_path: os.PathLike = Path(__file__).parent.parent.parent
    _local_pfun_share_path: os.PathLike = Path(os.path.expanduser("~/.local/share/pfun-cma-model"))
    _pfun_data_dirpath: os.PathLike = Path(os.path.join(os.path.abspath(pph.get_lib_path("pfun_common")), "data"))
    _remote_data_fpath: str = "https://github.com/pfun-health/pfun-data/releases/download/0.1.4/valid_data.csv"

    @property
    def _sample_data_fpath(self) -> Path:
        return PFunDataPath(
            raw_path=os.path.abspath(os.path.join(str(self._pfun_data_dirpath), "valid_data.csv")),
            sha256_digest="086622903c8f89cbcdafaa32128ae85f0088208b7dd1be0eafefb8306ad44abc",
        ).path

    def remove_sample_data(self) -> None:
        """Remove the sample data file if it exists."""
        if os.path.exists(self._sample_data_fpath):
            os.remove(self._sample_data_fpath)
            logger.debug("Sample data file %s removed.", self._sample_data_fpath)
        else:
            logger.warning(
                "(attempted to remove sample data) Sample data file %s does not exist." + " No action taken.",
                self._sample_data_fpath,
            )
        return

    def download_sample_data(self, overwrite: bool = False) -> None:
        """Download sample data from the remote file path."""
        if os.path.exists(self._sample_data_fpath) and not overwrite:
            logger.info("Sample data already exists at %s. Skipping download.", self._sample_data_fpath)
            return
        with httpx.Client(follow_redirects=True, max_redirects=2) as client:
            response = client.get(self._remote_data_fpath)
            if response.status_code == 200:
                with open(self._sample_data_fpath, "wb") as f:
                    f.write(response.content)
                logger.info("Sample data downloaded to %s.", self._sample_data_fpath)
            else:
                raise Exception(f"Failed to download sample data: {response.status_code}")

    def download_kaggle_brist1d(self, overwrite: bool = False) -> None:
        """Download the Brist1D dataset from Kaggle to the pfun_common/data/ directory and save as parquet using duckdb."""
        import zipfile
        import duckdb

        target_parquet_path = os.path.join(self._pfun_data_dirpath, "brist1d_train.parquet")
        if os.path.exists(target_parquet_path) and not overwrite:
            logger.info("Brist1D data already exists at %s. Skipping download.", target_parquet_path)
            return

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise ImportError("Kaggle API client not found. Please install the `kaggle` python package.")

        api = KaggleApi()
        api.authenticate()

        logger.info("Downloading brist1d competition dataset from Kaggle...")
        api.competition_download_file('brist1d', 'train.csv', path=str(self._pfun_data_dirpath))

        zip_path = os.path.join(self._pfun_data_dirpath, 'train.csv.zip')
        csv_path = os.path.join(self._pfun_data_dirpath, 'train.csv')

        # In case the api returns a zip file containing the csv
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract('train.csv', path=str(self._pfun_data_dirpath))
            os.remove(zip_path)

        if not os.path.exists(csv_path):
            # Fallback if download file wasn't zipped or named differently
            raise FileNotFoundError(f"Expected to find downloaded train.csv at {csv_path}")

        logger.info("Converting downloaded CSV to Parquet using DuckDB...")
        # Use DuckDB to convert the CSV to a Parquet file
        query = f"COPY (SELECT * FROM read_csv_auto('{csv_path}')) TO '{target_parquet_path}' (FORMAT PARQUET);"

        try:
            duckdb.sql(query)
            logger.info("Successfully converted and saved to %s.", target_parquet_path)
        except Exception as e:
            logger.error("Failed to convert CSV to Parquet using DuckDB: %s", str(e))
            raise
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path) # Clean up the original CSV

    def ensure_local_share_path_exists(self):
        """Create the pfun-cma-model local share path if it doesn't already exist."""
        pth = Path(self._local_pfun_share_path)
        if not pth.exists():
            return pth.mkdir(parents=True, exist_ok=True)
        logger.debug("pfun-cma-model local share path already exists (%s).", str(pth))

    @property
    def admin_db_fpath(self) -> str:
        return "sqlite+aiosqlite:///" + str(Path(self._local_pfun_share_path).joinpath("admin.db").absolute())

    @property
    def sample_data_fpath(self) -> Path:
        return Path(self._sample_data_fpath)

    @property
    def brist1d_data_fpath(self) -> Path:
        """Return the path to the downloaded brist1d_train.parquet file."""
        return Path(os.path.join(self._pfun_data_dirpath, "brist1d_train.parquet"))

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
