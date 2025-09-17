import os
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
import pfun_path_helper as pph  # type: ignore
pph.get_lib_path('pfun_cma_model')


__all__ = [
    'PFunDataPaths',
    'PFunAPIRoutes'
]


@dataclass
class PFunDataPaths:
    """Paths for data files used in the pfun_cma_model package."""

    _pfun_data_dirpath: os.PathLike = Path(
        os.path.abspath(pph.get_lib_path("pfun_data")))
    _sample_data_fpath: os.PathLike = Path(
        os.path.join(_pfun_data_dirpath, 'data/valid_data.csv'))

    @property
    def sample_data_fpath(self) -> Path:
        return Path(self._sample_data_fpath)

    @property
    def pfun_data_dirpath(self) -> Path:
        return Path(self._pfun_data_dirpath)

    def read_sample_data(self, fpath: Optional[os.PathLike] = None):
        """Read sample data from the specified file path."""
        if fpath is None:
            fpath = self.sample_data_fpath
        import pandas as pd
        return pd.read_csv(fpath)


@dataclass
class PFunAPIRoutes:
    FRONTEND_ROUTES = (
        '/run',
        '/run-at-time',
        '/params/schema',
        '/params/default'
    )

    PUBLIC_ROUTES = (
        '/',
        '/model/run',
        '/model/fit',
        '/model/run-at-time',
        '/params/schema',
        '/params/default',
    )

    PRIVATE_ROUTES = (
        ...
    )
