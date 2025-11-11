from pfun_cma_model.engine.calc import normalize_glucose
from pandas import (
    DataFrame,
    Series,
    DatetimeIndex,
    Timedelta,
    to_timedelta,
    to_datetime,
    isna,
    TimedeltaIndex
)
from numpy import array, nan, nansum, interp, ndarray
from numba import njit
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Union, Optional

root_path = str(Path(__file__).parents[1])
mod_path = str(Path(__file__).parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if mod_path not in sys.path:
    sys.path.insert(0, mod_path)

use_fastmath_global = False
njit_parallel = njit(cache=True, nogil=True,
                     fastmath=use_fastmath_global, parallel=True)
njit_serial = njit(cache=True, nogil=True, fastmath=use_fastmath_global)


def reindex_as_data(
    mdf: DataFrame,
    dindex: DatetimeIndex,
    dt: Timedelta
) -> DataFrame:
    """reindex a dataframe [mdf] to be like an index [dindex], use a tolerance [dt]"""
    return mdf.reindex(index=dindex, method="nearest", tolerance=dt)


def to_decimal_days(ixs: DatetimeIndex) -> ndarray:
    """convert DatetimeIndex -> array[float] (decimal days)"""
    return (ixs.to_series().apply(lambda ix: (ix.year * 365.0) + ix.day_of_year
                                  + (ix.hour / 24.0)).astype(float))


def to_decimal_hours(ixs: DatetimeIndex) -> ndarray:
    """convert DatetimeIndex -> array[float] (decimal hours)"""
    ixs_local = ixs.copy()
    if not isinstance(ixs, DatetimeIndex):
        ixs_local = DatetimeIndex(ixs_local)
    return array(
        [
            nansum([
                ix.year * 365.0 * 24.0,
                ix.day_of_year * 24.0,
                ix.hour,
                (ix.minute / 60.0),
                (ix.second / 3600.0),
            ]) for ix in ixs_local
        ],
        dtype=float,
    )


def to_decimal_secs(ixs: Union[DatetimeIndex, Series, List]) -> ndarray:
    secs = DatetimeIndex(
        ixs).to_series().diff().dt.total_seconds().cumsum().fillna(0.0)
    return 3600.0 * to_decimal_hours(ixs)[0] + secs


def dt_to_decimal_hours(dt: Timedelta) -> float:
    """convert Timedelta -> float (decimal hours)"""
    return nansum([
        dt.components.days * 24.0,
        dt.components.hours,
        (dt.components.minutes / 60.0),
        (dt.components.seconds / 3600.0),
    ])


def dt_to_decimal_secs(dt: Timedelta) -> float:
    """convert Timedelta -> float (decimal seconds)"""
    return nansum([
        dt.components.days * 24.0 * 3600.0,
        dt.components.hours * 3600.0,
        60.0 * dt.components.minutes,
        (dt.components.seconds),
    ])


def to_tod_hours(ixs: Union[DatetimeIndex, List, Series]) -> ndarray:
    """convert DatetimeIndex -> array[float] (decimal hours, [0.0, 23.99])"""
    return array(
        [
            float(tix.hour + (tix.minute / 60.0) + (tix.second / 3600.0))
            for tix in ixs
        ],
        dtype=float,
    )


@njit_serial
def _diff_tod_hours(tod0: float, tod1: float) -> float:
    return (12.0 - abs(abs(tod0 - tod1) - 12.0))


def diff_tod_hours(
    tod0: Union[ndarray, Series, float, int, List],
    tod1: Union[ndarray, Series, float, int, List]
) -> Union[float, ndarray]:
    """compute the absolute 'clock distance' between two time-of-day (decimal hours) arrays.
    """
    def _pre_conv(tod):
        if isinstance(tod, ndarray):
            return tod
        elif isinstance(tod, Series):
            tod = tod.to_numpy(dtype=float, na_value=nan)
        elif any([isinstance(tod, float), isinstance(tod, int)]):
            tod = float(tod)
        elif isinstance(tod, list):
            tod = array(tod, dtype=float)
        return tod
    tod0, tod1 = _pre_conv(tod0), _pre_conv(tod1)
    tod_diff = _diff_tod_hours(tod0, tod1)
    return tod_diff


def interp_missing_data(
    df: Union[DataFrame, Series],
    cols: Optional[List[str]] = None
) -> DataFrame:
    """
    Interpolates missing data in a DataFrame.

    Parameters:
        df (DataFrame): The DataFrame to interpolate missing data in.
        cols (list): Optional. A list of column names to interpolate missing data for. If not provided, all columns in the DataFrame will be used.

    Returns:
        DataFrame: The DataFrame with the missing data interpolated.
    """
    #: save the original index
    ix_original = df.index.copy()
    df = df.copy()
    if isinstance(df, Series):
        df = df.to_frame()
    if isinstance(df.index, DatetimeIndex):
        df.set_index(
            Series([float(ix.timestamp()) for ix in df.index], dtype=float),
            inplace=True,
        )
    elif isinstance(df.index, TimedeltaIndex):
        df.set_index(
            Series([float(dt_to_decimal_secs(ix))
                   for ix in df.index], dtype=float),
            inplace=True,
        )
    if not isinstance(df.index[0], float):
        raise RuntimeError(f"df.index (currently: {type(df.index[0])}) "
                           "must be integer type (see `Timestamp.value()`)!")
    if cols is None:
        cols = []
    if len(cols) == 0:
        cols = list(df.columns)
    for col in cols:
        xvals = df[col].index[df[col].apply(lambda row: isna(row))]
        if len(xvals) == 0:
            continue
        other_ixs = [ix for ix in df.index if ix not in xvals]
        df.loc[xvals, col] = interp(
            xvals, other_ixs, df.loc[other_ixs, col])  # type: ignore
    df.set_index(ix_original, inplace=True)
    return df


def downsample_data(
    df: Union[DataFrame, Series],
    N: int = 1024
) -> DataFrame | Series:
    """
    Downsamples the given DataFrame to obtain N (default=1024) samples.

    Parameters:
    - df (DataFrame): The DataFrame to be downsampled.
    - N (integer): Number of samples in the result.

    Returns:
    - df (DataFrame): The downsampled DataFrame with 'N' timesteps.
    """
    #: end up with N samples
    freq = Timedelta(hours=(df.index.max() - df.index.min()
                            ).total_seconds() / 3600) / (N - 1)
    df = df.resample(freq).mean()
    return df


def format_data(
    records: Union[Dict, DataFrame],
    N: int = 1024,
    tz_offset: Optional[Union[int, float]] = None
) -> DataFrame:
    """Format data for the model.

    Notes:
    - The `ts_utc` and `ts_local` columns are converted to `systemTime` and `displayTime` columns.
    - Glucose will be normalized to [0.0, 2.0], where 0.0 is very low and 2.0 is very high. This is done by calling the `normalize_glucose` function from the `pfun_cma_model.engine.calc` module.
    - If `tz_offset` is not provided, it will be computed from the `ts_local` column.
    - The output will contain exactly 1024 samples (corresponding to 1.40625-minute intervals for a single 24-hour period). Chosen for performance reasons, as the integer indices of the output will contain exactly 4096 bytes. This is done by calling the `downsample_data` function from the `pfun_cma_model.engine.data_utils` module.

    Parameters:
    - records (Dict | DataFrame): The records to be formatted.
    - N (integer): Desired number of samples in the result (default=1024).
    - tz_offset (None | int | float): The timezone offset in decimal seconds.

    Returns:
    - df (DataFrame): The formatted data.
    """
    if isinstance(records, dict):
        df = DataFrame.from_records(records)
    else:
        df = records.copy()
    if not any([col in df.columns for col in ["ts_utc", "ts_local", "time"]]):
        raise RuntimeError(
            "No raw time column ('ts_utc', 'ts_local', 'time') was present in the provided dataframe...\n"
            "Perhaps this data has already been formatted?"
        )
    if not any(['ts_utc' in df.columns, 'ts_local' in df.columns]):
        df['ts_local'] = to_datetime(df['time'], utc=False, format="ISO8601")
    if 'ts_utc' not in df.columns:
        df['ts_utc'] = df['ts_local'].dt.tz_convert('UTC')
    if "systemTime" not in df.columns:
        df["systemTime"] = df["ts_utc"]
    if "displayTime" not in df.columns:
        df["displayTime"] = df["ts_local"]
    #: input time (utc, tz-aware)
    time = to_datetime(df["systemTime"], utc=True, format="ISO8601")
    #: ! important to leave systemTime and displayTime un-localized
    # #: ...to later compute tz_offset.
    df["systemTime"] = to_datetime(df["systemTime"])
    df["displayTime"] = to_datetime(df["displayTime"])
    #: compute the tz_offset if not given
    if tz_offset is None:
        try:
            tz_offset = df["displayTime"].iloc[0].utcoffset().total_seconds()
        except AttributeError:
            tz_offset = (df["displayTime"].iloc[0] -
                         df["systemTime"].iloc[0]).total_seconds() / 3600.0
    #: offset using tz_offset
    time = time + Timedelta(hours=tz_offset)  # type: ignore
    df["tod"] = to_tod_hours(time)
    df["t"] = to_timedelta(df["tod"], unit="h")
    df["t"] = df["t"].apply(lambda d: dt_to_decimal_hours(d)).astype(float)
    if "value" not in df.columns:
        df["value"] = df["sg"]  # for practice data
    gvalues = df["value"].to_numpy(dtype=float, na_value=nan)
    gvalues_normed = normalize_glucose(gvalues)
    df["G"] = gvalues_normed
    df["time"] = time
    df.sort_values(by="time", inplace=True)
    keepers = ["time", "value", "tod", "t", "G"]
    df = df[keepers]
    #: reindex to have consistent sampling (default is 1024 samples, correspond to bytes)
    df = df.set_index("time", drop=True).sort_index()
    df = downsample_data(df, N=N)  # type: ignore
    df.reset_index(names="time", inplace=True)  # type: ignore
    df = df.set_index("time", drop=True).sort_index()
    #: interpolate missing data
    df = interp_missing_data(df)
    df.reset_index(names="time", inplace=True)
    df.set_index("t", inplace=True, drop=False)
    df.sort_index(inplace=True)
    return df
