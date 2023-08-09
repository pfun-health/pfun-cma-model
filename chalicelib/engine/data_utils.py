import pandas as pd
import numpy as np
import importlib
import sys
from pathlib import Path
from typing import (
    Dict
)

root_path = str(Path(__file__).parents[1])
mod_path = str(Path(__file__).parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if mod_path not in sys.path:
    sys.path.insert(0, mod_path)
normalize_glucose = importlib.import_module(
    ".calc", package="chalicelib.engine").normalize_glucose

def reindex_as_data(mdf, dindex, dt):
    """reindex a dataframe [mdf] to be like an index [dindex], use a tolerance [dt] 
    """
    return mdf.reindex(index=dindex, method='nearest', tolerance=dt)


"""Convert between datetime representations:
"""


def to_decimal_days(ixs: pd.DatetimeIndex):
    """convert pd.DatetimeIndex -> np.array[float] (decimal days)
    """
    return ixs.to_series().apply(lambda ix: (ix.year * 365.0) + ix.day_of_year
                                 + (ix.hour / 24.0)).astype(float)


def to_decimal_hours(ixs: pd.DatetimeIndex):
    """convert pd.DatetimeIndex -> np.array[float] (decimal hours)
    """
    ixs_local = ixs.copy()
    if not isinstance(ixs, pd.DatetimeIndex):
        ixs_local = pd.DatetimeIndex(ixs_local)
    return np.array([
        np.nansum([
            ix.year * 365.0 * 24.0, ix.day_of_year * 24.0,
            ix.hour, (ix.minute / 60.0), (ix.second / 3600.0)]) for ix in
        ixs_local], dtype=float)


def to_decimal_secs(ixs):
    secs = pd.DatetimeIndex(ixs).to_series().diff(
    ).dt.total_seconds().cumsum().fillna(0.0)
    return 3600.0 * to_decimal_hours(ixs)[0] + secs


def dt_to_decimal_hours(dt):
    """convert pd.Timedelta -> float (decimal hours)
    """
    return np.nansum([dt.components.days * 24.0, dt.components.hours,
                      (dt.components.minutes / 60.0), (dt.components.seconds / 3600.0)])


def dt_to_decimal_secs(dt):
    """convert pd.Timedelta -> float (decimal seconds)
    """
    return np.nansum([dt.components.days * 24.0 * 3600.0,
                      dt.components.hours * 3600.0, 60.0 * dt.components.minutes,
                      (dt.components.seconds)])


def to_tod_hours(ixs):
    """convert pd.DatetimeIndex -> np.array[float] (decimal hours, [0.0, 23.99])
    """
    return np.array([float(tix.hour + (tix.minute / 60.0) + (tix.second / 3600.0))
                     for tix in ixs], dtype=float)


def interp_missing_data(df: pd.DataFrame, cols: list = []):
    """
    Interpolates missing data in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to interpolate missing data in.
        cols (list): Optional. A list of column names to interpolate missing data for. If not provided, all columns in the DataFrame will be used.

    Returns:
        pd.DataFrame: The DataFrame with the missing data interpolated.
    """
    #: save the original index
    ix_original = df.index.copy()
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df.set_index(pd.Series([float(ix.timestamp()) for ix in df.index],
                               dtype=float), inplace=True)
    if not isinstance(df.index[0], (float, np.float64)):
        raise RuntimeError(f'df.index (currently: {type(df.index[0])}) '
                           'must be integer type (see `pd.Timestamp.value()`)!')
    if len(cols) == 0:
        cols = list(df.columns)
    for col in cols:
        xvals = df[col].index[df[col].apply(lambda row: pd.isna(row))]
        if len(xvals) == 0:
            continue
        other_ixs = [ix for ix in df.index if ix not in xvals]
        df.loc[xvals, col] = np.interp(xvals, other_ixs, df.loc[other_ixs, col])
    df.set_index(ix_original, inplace=True)
    return df


def format_data(records: Dict | pd.DataFrame, tz_offset: None | int | float = None) -> pd.DataFrame:
    """ready data for the model"""
    if isinstance(records, dict):
        df = pd.DataFrame.from_records(records)
    else:
        df = records.copy()
    if "systemTime" not in df.columns:
        df['systemTime'] = df['ts_utc']
    if "displayTime" not in df.columns:
        df['displayTime'] = df['ts_local']
    #: input time (utc, tz-aware)
    time = pd.to_datetime(df["systemTime"], utc=True, format='ISO8601')
    #: ! important to leave systemTime and displayTime un-localized
    # #: ...to later compute tz_offset.
    df["systemTime"] = pd.to_datetime(df["systemTime"])
    df["displayTime"] = pd.to_datetime(df["displayTime"])
    #: compute the tz_offset if not given
    if tz_offset is None:
        try:
            tz_offset = df["displayTime"].iloc[0] \
                .utcoffset().total_seconds()
        except AttributeError:
            tz_offset = (df['displayTime'].iloc[0] - df['systemTime'].iloc[0]) \
                .total_seconds() / 3600.0
    #: offset using tz_offset
    time = time + pd.Timedelta(hours=tz_offset)
    df['tod'] = to_tod_hours(time)
    df["t"] = pd.to_timedelta(df['tod'], unit='H')
    df["t"] = df["t"].apply(lambda d: dt_to_decimal_hours(d)).astype(float)
    if "value" not in df.columns:
        df["value"] = df["sg"]  # for practice data
    gvalues = df["value"].to_numpy(dtype=float, na_value=np.nan)
    gvalues_normed = normalize_glucose(gvalues)
    df["G"] = gvalues_normed
    df["time"] = time
    df.sort_values(by="time", inplace=True)
    keepers = ["time", "value", "tod", "t", "G"]
    df = df[keepers]
    #: reindex to have consistent sampling before rounding...
    df = df.set_index('time', drop=True).sort_index()
    df = df.resample('30T').mean()
    df.reset_index(names='time', inplace=True)
    #: downsample relative to rounded TOD (decimal hours)
    df["tod"] = df["tod"].apply(lambda d: np.round(d, 3))
    df = df.groupby("tod", as_index=False).mean()
    df = df.set_index('time', drop=True).sort_index()
    #: interpolate missing data
    df = interp_missing_data(df)
    df.reset_index(names='time', inplace=True)
    df.set_index("t", inplace=True, drop=False)
    df.sort_index(inplace=True)
    return df
