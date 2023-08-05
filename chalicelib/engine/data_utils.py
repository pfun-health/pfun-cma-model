import pandas as pd
import numpy as np


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
