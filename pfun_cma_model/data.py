from pfun_cma_model.engine.data_utils import format_data
from pfun_cma_model.misc.pathdefs import PFunDataPaths
import pandas as pd

__all__ = [
    "format_data",
    "PFunDataPaths",
    "read_sample_data",
]


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
    return df.to_json(orient='records')
