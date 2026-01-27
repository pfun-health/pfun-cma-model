from pathlib import Path
import pandas as pd

from pfun_cma_model.engine.data_utils import format_data
from pfun_cma_model.misc.pathdefs import PFunDataPaths

__all__ = [
    "format_data",
    "PFunDataPaths",
    "read_sample_data",
    "get_db_path",
]


def get_db_path() -> Path:
    """Return the database directory path."""
    from pfun_cma_model.misc.pathdefs import PFunDataPaths
    data_dirpath = Path(PFunDataPaths().pfun_data_dirpath)
    return data_dirpath / "pfun_cma_model.db"


def read_sample_data(convert2json: bool) -> pd.DataFrame | str:
    """Read the sample dataset from the PFunDataPaths.

    Args:
        convert2json (bool): If True, convert the DataFrame to JSON string format.

    Returns:
        pd.DataFrame or str: The sample dataset as a DataFrame or JSON string.

    Example:
        >>> df = read_sample_data()
        >>> print(df.head())

    """
    from pfun_cma_model.misc.pathdefs import PFunDataPaths

    df = PFunDataPaths().read_sample_data()
    if convert2json is False:
        return df
    return df.to_json(orient="records")
