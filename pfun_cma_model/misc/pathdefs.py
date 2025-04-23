import os
from dataclasses import dataclass
import pfun_path_helper as pph

__all__ = [
    'PFunDataPaths',
    'PFunAPIRoutes'
]


@dataclass
class PFunDataPaths:
    """Paths for data files used in the pfun_cma_model package."""

    pfun_data_dirpath = os.path.abspath(pph.get_lib_path("pfun_data"))
    sample_data_fpath = os.path.join(pfun_data_dirpath, 'data/valid_data.csv')

    def read_sample_data(self):
        """Read sample data from the specified file path."""
        import pandas as pd
        return pd.read_csv(self.sample_data_fpath)


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
        '/run',
        '/fit',
        '/run-at-time',
        '/routes',
        '/sdk',
        '/params/schema',
        '/params/default',
    )

    PRIVATE_ROUTES = (
        '/run',
        '/fit',
        '/run-at-time',
        '/sdk'
    )