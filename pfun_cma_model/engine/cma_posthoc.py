import logging
from argparse import Namespace as Namespace_
import numpy as np
from dataclasses import dataclass
from typing import Any
from pfun_cma_model.engine.fit import CMAFitResult
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Namespace(Namespace_):
    """wrapper around argparse.Namespace that implements a simple __getitem__ method.
    Examples:
    ---------
    >>> ns = Namespace(a=1, b=2)
    >>> ns['a']
    1
    >>> ns['b']
    2
    """

    def __getitem__(self, key):
        assert key in self.__dict__, f"key {key} not found in Namespace"
        return self.__dict__[key]


def calc_model_stats(cma):
    """
    Calculates the model statistics based on the given `cma` object.

    Parameters:
        cma (CMASleepwakeModel): The CMASleepwakeModel object containing the data.

    Returns:
        dict: A dictionary containing the calculated model statistics with the following keys:
            - "G_morn" (float): The average value of `g_morning` in the `cma` object.
            - "G_eve" (float): The average value of `g_evening` in the `cma` object.
            - "I_S_morn" (float): The average value of `I_morning` in the `cma` object.
            - "I_S_eve" (float): The average value of `I_evening` in the `cma` object.
    """
    stats = {
        "g_morn": float(np.nanmean(cma.g_morning)),
        "g_eve": float(np.nanmean(cma.g_evening)),
        "i_s_morn": float(np.nanmean(cma.I_morning)),
        "i_s_eve": float(np.nanmean(cma.I_evening))
    }
    return stats


@dataclass
class ChronometabolicIndex:
    cma_fit_result: CMAFitResult
    _columns: tuple = ("c", "m", "a", "g")

    def __post_init__(self):
        pass

    @classmethod
    def find_peaks_and_troughs(cls, df: pd.DataFrame, inplace=True):
        """
        Find peaks and troughs in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            inplace (bool, optional): If True, modify the DataFrame in-place. Defaults to True.

        Returns:
            pd.DataFrame: The modified DataFrame with peaks and troughs labeled.
        """
        # compute the peaks and troughs for c, m, a, g.
        if inplace is False:
            df = df.copy()
        for column in cls._columns:
            peaks = df[df[column] > df[column].shift(1) & df[column] > df[column].shift(-1)]
            troughs = df[df[column] < df[column].shift(1) & df[column] < df[column].shift(-1)]
            data_agg = [
                (f"{column}_peaks", peaks, pd.Series.idxmax),
                (f"{column}_troughs", troughs, pd.Series.idxmin)
            ]
            for label, data, oper in data_agg:
                df[label] = data.astype(int)  # label local peaks and troughs with 1
                global_extreme = oper(df[label])
                df[label].iat[global_extreme] = 2  # label global extrema with 2
        return df

    @classmethod
    def calc_cmi(cls, df: pd.DataFrame):
        pass


@dataclass
class ModelResultStats:
    G_morn: float
    G_eve: float
    I_S_morn: float
    I_S_eve: float