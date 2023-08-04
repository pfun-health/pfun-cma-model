import itertools
import json
import logging
import os
import pathlib
import pickle
import traceback
import traceback as tb
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property as _cached_property
from functools import wraps
from pathlib import Path
from pickle import UnpicklingError
from typing import AnyStr, Container, Literal, Union

import numba
import numpy as np
import pandas as pd
from numba import float64, guvectorize, jit, njit, prange, vectorize
from pandas.tseries.frequencies import to_offset
from scipy.interpolate import (Akima1DInterpolator, CubicSpline,
                               PchipInterpolator, PPoly)
from scipy.stats import variation

logging.captureWarnings(True)


__all__ = [
    'fuzzy_argwhere_close', 'inverse_sort', 'DupeColRenamer', 'get_nearest_time',
    'auto_pct_change', 'RiskComputer', 'make_extra_tags', 'make_time_of_day_column',
    'reindex_as_data', 'time_gradient', 'PFunDayNight', 'compute_confusion_matrix',
    'calc_mcc', 'calc_accuracy', 'calc_sensitivity', 'diff_tod_hours',
    'to_decimal_hours', 'dt_to_decimal_hours', 'periodic_interp', 'normalize',
    'to_tod_hours', 'dt_to_decimal_secs', 'to_decimal_secs', 'find_turning_points',
    'analyze_spike_patterns', 'calc_ratio_5pct', 'is_high', 'is_low', 'expit_numba',
    'normed_absdiff_1d', 'normed_absdiff_1d',
    'max_numba_ax', 'max_numba', 'min_numba', 'sum_numba',
    'argsort_numba', 'weighted_avg_numba', 'weighted_avg_numba_serial',
    'check_denom', 'GlobalConst', 'calc_err', 'interps', 'InterpKindOption',
    'TaggedCacheMixin', 'tagged_property', 'numba_softmax', 'softmax_numba',
    'njit_parallel'
]


class GlobalConst:

    """Global constants:
    """

    #: a (very) small number
    EPS = 1e-13

    #: Conway's constant
    CONWAY = 1.303577269034296391257099112152551890730702504659404875754861390628550

    #: Pi^4
    PI4 = 97.40909103400243723644033268870511124972758567268542169146785938997085

    #: Dirichlet L(4,chi) [ ref : https://tinyurl.com/dirichletL4 ]
    DIRL4 = PI4 / 96.0

    #: sqrt(2) (1.4142135623730951...)
    SQRT2 = np.sqrt(2.0)

    #: Gieseking-Konstante (1.0149416064096537...)
    GIEKONST = 1.01494160640965362502

    #: Phi (golden ratio)
    PHI = 1.61803398874989484820458683436563811772030917980576286213544862270526046281890244970720720418939113748475

    #: GCNO scale constant
    GCNO = 2.77


class tagged_property(_cached_property):
    """extension of functools.cached_property to include a 'tag' value

    ref: https://stackoverflow.com/a/41938579/1871569
    """

    def __init__(self, func, *tags):
        super().__init__(func)
        self.tags = frozenset(tags)

    @classmethod
    def tag(cls, *tags):
        return lambda f: cls(f, *tags)


class TaggedCacheMixin:
    """easily invalidate "tagged" properties (functools.cached_property -> with tags)

    ref: https://stackoverflow.com/a/41938579/1871569
    """

    def invalidate_all(self):
        for key, value in self.__class__.__dict__.items():
            if isinstance(value, tagged_property):
                self.__dict__.pop(key, None)

    def reset_cache(self):
        #: alias for `self.invalidate_all`
        self.invalidate_all()

    def invalidate(self, tag: str):
        """invalidate a single cached property
        """
        for key, value in self.__class__.__dict__.items():
            if isinstance(value, tagged_property) and tag in value.tags:
                self.__dict__.pop(key, None)


"""numba macros/wrappers:
"""

use_fastmath_global = False
njit_parallel = numba.njit(cache=True, nogil=True,
                           fastmath=use_fastmath_global, parallel=True)
njit_serial = numba.njit(cache=True, nogil=True, fastmath=use_fastmath_global)


"""numba-optimized data utility functions:
"""


@guvectorize(['void(float64[:], intp[:], float64[:])'],
             '(n),()->(n)')
def move_mean_numba(a, window_arr, out):
    """compute the windowed moving average
    (handles edge cases correctly)

    Args:
        a (np.ndarray): 1-D numpy array
        window_arr (int): window length (stored as a pointer)

    Returns:
        np.ndarray: windowed moving average result as an array; same shape as `a`.
    """
    window_width = window_arr[0]
    asum = 0.0
    count = 0
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / count


@njit_serial
def calc_err(xm, xd):
    """calculate the objective function (residual error)
    between the model solution (`xm`) & the data (`xd`).

    Args:
        xm (np.ndarray): model solution vector
        xd (np.ndarray): input data vector

    Returns:
        float: total residual error
    """
    if np.any(xm < 0):
        return np.inf
    if xd.size > xm.size:
        xd = xd[:xm.size]
    elif xm.size > xd.size:
        xm = xm[:xd.size]
    mad = np.abs(xm - xd)
    mad = np.nanmean(mad / xd)
    xmin, xmax = np.nanmin(xd), np.nanmax(xd)
    dmin = np.abs(np.nanmin(xm) - xmin) / xmin
    dmax = np.abs(np.nanmax(xm) - xmax) / xmax
    xstd = np.nanstd(xd)
    dstd = np.abs(np.nanstd(xm) - xstd) / xstd
    resid = (mad + dmin + dmax + dstd) / 4.0
    return resid


@jit("float64(float64[:])", cache=True, nopython=True, nogil=True, parallel=False)
def esum(z):
    return np.sum(np.exp(z))


@jit("float64[:](float64[:])", cache=True, nopython=True, nogil=True, parallel=False)
def _numba_softmax(z):
    """ref: https://alexpnt.github.io/2018/10/19/speeding-up-python/
    """
    num = np.exp(z)
    s = num / esum(z)
    return s


def numba_softmax(z, norm_first=True):
    z = z.ravel()
    if norm_first is True:
        z = normalize(z)
    result = _numba_softmax(z)
    return result


def softmax_numba(z, norm_first=True):
    return numba_softmax(z, norm_first=True)


@vectorize([float64(float64)], fastmath=False, nopython=True)
def _expit_numba(xx):
    return 1.0 / (1.0 + np.exp(-2.0 * xx))


@njit_serial
def expit_numba(xx):
    return _expit_numba(xx)


@njit_serial
def _normalize(x, a, b):
    """normalize a flattened float array between a and b
    """
    xmin, xmax = x.min(), x.max()
    return a + (b - a) * (x - xmin) / (xmax - xmin)


def normalize(x, a: float = 0.0, b: float = 1.0):
    """normalize a flat 1-d np.ndarray[float] between a and b
    """
    if isinstance(x, pd.Series):
        x = x.to_numpy(dtype=float, na_value=np.nan)
    assert x.ndim < 2
    x = np.array(x, dtype=float).flatten()
    return _normalize(x, a, b)


@njit(float64[:](float64[:], float64[:]), fastmath=False, cache=True, parallel=False)
def normed_absdiff_1d(v1, v2):
    """normed_absdiff_1d: compute the max-normalized absolute difference
    between two 1-D vectors.

    *note*: `np.max(v1)` is used as the norm.
    """
    assert v1.ndim < 2
    norm = np.max(v1)
    return np.abs(v1 - v2) / norm


@njit_serial
def max_numba_ax(x, axis: int = 1):
    """numba-optimized method to replicate: 
        `np.max(x, axis=1)`
    ref: https://numba.discourse.group/t/numba-performance-doesnt-scale-as-well-as-numpy-in-vectorized-max-function/782/14
    """
    if axis == 0:
        x = x.transpose()
    res = np.full((x.shape[0],), -np.inf, dtype=x.dtype)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp = x[i, j]
            if res[i] < tmp:
                res[i] = tmp
    return res


@njit(fastmath=False, cache=True, parallel=False)
def normed_absdiff_2d(v1, v2, axis: int = 0):
    """ v1: 2d array [N, M] (long, short)
        v2: 1d array [N, 1]
        return: 1d array [N, 1]
    """
    assert v1.ndim == 2
    assert v2.ndim < 2
    other_ax = v1.shape[0 if axis == 1 else 1]
    res = np.full(v1.shape[axis], 0.0, dtype=np.float64)
    norm = max_numba_ax(v1, axis=other_ax)  #: norm.shape = [M, ]
    for i in range(v1.shape[axis]):
        res[i] = np.sum(np.abs(v1[i, :] - v2[i]) / (1.0 + norm[i]))
    res /= float(v1.shape[axis])
    return res


@njit_serial
def sum_numba(x, axis: int = None):
    if axis is None:
        return np.sum(x, dtype=x.dtype)
    return np.sum(x, axis=axis, dtype=x.dtype)


@njit_serial
def max_numba(x):
    return np.max(x)


@njit_serial
def min_numba(x):
    return np.min(x)


@njit(fastmath=False, cache=True, nogil=True)
def argsort_numba(x):
    return np.argsort(x)


@njit("float64[:](float64[:, :],float64[:])", fastmath=False, cache=True, nogil=True, parallel=True)
def weighted_avg_numba(x, w):
    accum = np.zeros(x.shape[1])
    wnorm = w.sum()
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            accum[j] += (x[i, j] * w[i])
    accum /= wnorm
    return accum


weighted_avg_numba_serial =\
    njit("float64[:](float64[:, :],float64[:])", fastmath=False, cache=True, nogil=True,
         parallel=False)(weighted_avg_numba.py_func)


@njit_serial
def calc_ratio_5pct(y_true, y_pred):
    """compute the ratio of y_pred,
    s.t. (|y_pred - y_true| / y_pred) <= 5% err.
    """
    abs_err = np.abs(y_true - y_pred) / y_true
    is_lt_5pct = (abs_err <= 0.05).astype(numba.float64)
    ratio = np.nanmean(is_lt_5pct)
    return ratio


"""Find highs/lows/spikes:
"""


def ensure_np_float(f):
    """decorator to ensure the input is converted to a flattened np.ndarray[float]"""
    @wraps(f)
    def wrapper(*args):
        new_args = []
        for a in args:
            if isinstance(a, Container) and not isinstance(a, np.ndarray):
                a = np.array(a, dtype=float)
            new_args.append(a)
        return f(*new_args)
    return wrapper


@numba.vectorize([numba.bool_(numba.float64, numba.float64)], fastmath=False)
def _is_high(x, thresh_high):
    return np.logical_and(x >= thresh_high, np.isfinite(x))


@ensure_np_float
def is_high(x, thresh_high: float):
    """find highs (>= `thresh_high`)
    """
    return _is_high(x, thresh_high)


@numba.vectorize([numba.bool_(numba.float64, numba.float64)], fastmath=False)
def _is_low(x, thresh_low):
    return np.logical_and((x <= thresh_low), (np.isfinite(x)))


@ensure_np_float
def is_low(x, thresh_low: float):
    """find lows (<= `thresh_low`)
    """
    return _is_low(x, thresh_low)


@njit([float64(float64[:])], fastmath=False, nogil=True, cache=True)
def _calc_range(x):
    return np.max(x) - np.min(x)


def calc_range(x):
    x = np.array(x, dtype=float)
    return _calc_range(x)


@njit(cache=True, nogil=True)
def check_denom(denom):
    if np.isnan(denom) or denom == 0.0:
        return -1.0
    return float(denom)


"""non-njitted data utils:
"""


def find_turning_points(vec, thresh):
    """return indices where `vec` was previously >= `thresh`,
    ...and now (`vec.loc[vec.index + dt]`) is < `thresh`.

    vec: pd.Series[float] (indices are `pd.DatetimeIndex`)
    thresh: float
    return: 
    """
    return vec.apply(lambda d: d >= thresh).astype(float).diff().loc[lambda d: d == -1].index


def reindex_as_data(mdf, dindex, dt):
    """reindex a dataframe [mdf] to be like an index [dindex], use a tolerance [dt] 
    """
    return mdf.reindex(index=dindex, method='nearest', tolerance=dt)


def analyze_spike_patterns(vec, vmin, vmax):
    """compute high/low spikes, and analyze temporal spike patterns

    vec: pd.Series[float] (indices are `pd.DatetimeIndex`)
    vmin: float: min threshold
    vmax: float: max threshold
    return: pd.DataFrame
    """
    index_col = vec.index.name
    df = pd.DataFrame({'vec': vec, index_col: vec.index}, index=vec.index)
    df['hi'] = is_high(vec.to_numpy(dtype=float, na_value=np.nan), vmax)
    df['lo'] = is_low(vec.to_numpy(dtype=float, na_value=np.nan), vmin)
    df['spikes'] = np.logical_or(df['hi'], df['lo'])
    df['ts_dur'] = None
    # group into consecutive events
    df['spk_group'] = (df['spikes'].diff(1) != 0).cumsum()
    # compute event durations
    nagg = pd.NamedAgg(column=index_col, aggfunc=lambda g: g.max() - g.min())
    df_durs = df.groupby(['spikes', 'spk_group'],
                         as_index=False).agg(ts_dur=nagg)
    ts_durs = df.apply(
        lambda d: df_durs.loc[df_durs['spk_group'] == d['spk_group'], 'ts_dur'].item(), axis=1)
    df['ts_dur'] = ts_durs
    df['ts_dur_secs'] = df.ts_dur.apply(lambda d: dt_to_decimal_secs(d))
    df['ts_dur_hrs'] = df.ts_dur.apply(lambda d: dt_to_decimal_hours(d))
    if (df.spikes == False).all():
        df['ts_dur_normed'] = np.ones((df.shape[0],))
        return df
    ts_dur_normed = df.groupby(['spikes']).apply(
        lambda g: g['ts_dur_secs'].to_numpy(dtype=float, na_value=np.nan)).apply(normalize)
    if (df.spikes == True).any():
        df.loc[df.spikes == True, 'ts_dur_normed'] = ts_dur_normed[True]
    if (df.spikes == False).any():
        df.loc[df.spikes == False, 'ts_dur_normed'] = ts_dur_normed[False]
    return df


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


@njit_serial
def _diff_tod_hours(tod0, tod1):
    return (12.0 - np.abs(np.abs(tod0 - tod1) - 12.0))


def diff_tod_hours(tod0, tod1):
    """compute the absolute 'clock distance' between two time-of-day (decimal hours) arrays.
    """
    def _pre_conv(tod):
        if isinstance(tod, np.ndarray):
            return tod
        elif isinstance(tod, pd.Series):
            tod = tod.to_numpy(dtype=float, na_value=np.nan)
        elif any([isinstance(tod, float), isinstance(tod, int)]):
            tod = float(tod)
        elif isinstance(tod, list):
            tod = np.array(tod, dtype=float)
        return tod
    tod0, tod1 = _pre_conv(tod0), _pre_conv(tod1)
    tod_diff = _diff_tod_hours(tod0, tod1)
    return tod_diff


# interpolator types (see `interp_kind` keyword arg for `periodic_interp`)
interps = {'akima': Akima1DInterpolator,
           'pchip': PchipInterpolator, 'cubic': CubicSpline}
InterpKindOption = Literal['akima', 'pchip', 'cubic', 'numpy', 'none']


def periodic_interp(vec: pd.Series, original_index: pd.Index, dt=None, dt_window_size='auto',
                    dt_period_scale=2.0*np.pi, correction_weight=None, normalize_output=False,
                    only_nonfinite=True, interp_kind: InterpKindOption = 'none', ret_interp=False) -> pd.Series:
    """Periodic interpolation (see `samples/experiment_with_ewm.py`)

    [args]
        vec: pd.Series
        original_index: pd.DatetimeIndex
        dt: pd.Timedelta (*Required if `interp_kind='numpy'`)
        dt_window_size: int : rolling avg window size (dt * dt_window_size); 
            If 'auto', estimate from the space between non-nan indices.
            (*Only used if `use_pchip=False`)
        dt_period_scale: float : used as `np.interp(..., period=dt_period_scale*dt_to_decimal_hours(dt))`
            (*Only used if `use_pchip=False`)
        correction_weight: float : re-weight as sum([correction_weight*vec, interp_vec]) / (correction_weight + 1.0)
        normalize_output: bool: if True, normalize the interpolated output so that the range matches `vec` before returning.
        only_nonfinite: bool: if True (default), only use the interpolated values for missing/non-finite values in the input `vec`.
        interp_kind: str: choose one of {'akima' (default), 'pchip', 'cubic', 'ppoly', 'numpy'}
        ret_interp: bool: if True, return the interpolator instance (unless `interp_kind='numpy'`), along with the interpolated series.
    """
    global interps
    original_index = pd.DatetimeIndex(original_index)
    vec_npy = vec.to_numpy(na_value=np.nan, dtype=float)

    def find_missing(v):
        return np.logical_or(np.logical_not(np.isfinite(v)), pd.isna(v))
    interp_inst = None
    interp_kind = interp_kind.lower()
    # ! interpolate using `np.interp` (if `use_pchip=False`)
    if interp_kind == 'numpy':
        try:
            dt = pd.Timedelta(dt)
        except ValueError:
            dt = pd.Timedelta(pd.DatetimeIndex(
                original_index).to_series().diff().dropna().min())
        if dt_window_size == 'auto':
            dt_window_size = int(np.ceil(pd.DatetimeIndex(
                vec[np.isfinite(vec_npy)].index).to_series().diff().dropna().mean() / dt))
        vec_res = vec.dropna().rolling(dt * dt_window_size,
                                       center=True).mean().dropna()  # type: ignore
        arr_res = vec_res.to_numpy(dtype=float, na_value=np.nan)
        period = dt_period_scale*dt_to_decimal_hours(dt)
        dechours_index = to_decimal_hours(vec_res.index)
        vout = np.interp(original_index, dechours_index,
                         arr_res, period=period)
    elif interp_kind == 'none':  # ! no interpolation, return original series
        vout = vec_npy
        vout = pd.Series(vout).bfill().ffill().to_numpy(dtype=float)
    else:  # ! interpolate using another type of interpolator, e.g. akima
        gixs = original_index[np.logical_not(find_missing(vec))]
        interp_inst = interps[interp_kind](gixs, vec[gixs])
        # ! important to use periodic extrapolation
        vout = interp_inst(original_index, extrapolate='periodic')
    if correction_weight is not None:
        vout = np.nansum([correction_weight * vec_npy, vout],
                         axis=0) / (correction_weight + 1.0)  # type: ignore
    if normalize_output is True:
        vout = normalize(vout, a=float(vec_npy.min()),
                         b=float(vec_npy.max()))  # type: ignore
    vout = pd.Series(vout, index=original_index, dtype=float)  # type: ignore
    if only_nonfinite is True:
        vout = vec.where(np.isfinite(vec_npy), vout, inplace=False)
    if ret_interp is False:
        return vout
    return (vout, interp_inst)


"""Contingency metrics:
"""


def calc_sensitivity(cmat, which):
    if isinstance(which, str):
        denom = sum([
            cmat[f'{which}_TP'],
            cmat[f'{which}_FN']
        ])
        denom = check_denom(denom)
        return float(cmat[f'{which}_TP'] / denom)
    return {w: calc_accuracy(cmat, w) for w in which}


def calc_accuracy(cmat, which):
    if isinstance(which, str):
        denom = sum((cmat[f'{which}_TP'],
                     cmat[f'{which}_TN'],
                     cmat[f'{which}_FP'],
                     cmat[f'{which}_FN']))
        denom = check_denom(denom)
        numer = (cmat[f'{which}_TP'] + cmat[f'{which}_TN'])
        return float(numer / denom)
    return {w: calc_accuracy(cmat, w) for w in which}


def calc_mcc(cmat, keys=['lows', 'highs', 'spikes'], as_numpy=False):
    '''
    compute the Matthew Correlation Coefficient (MCC) from
    the confusion matrix `cmat` (as returned from `compute_confusion_matrix`).

    returns a dictionary unless `as_numpy` flag is True.
    '''
    if isinstance(keys, str):
        keys = [keys, ]
    mcc_dict = OrderedDict()
    for key in keys:
        numer = (
            (cmat[f'{key}_TP'] * cmat[f'{key}_TN']) -
            (cmat[f'{key}_FP'] * cmat[f'{key}_FN'])
        )
        denom = np.sqrt((cmat[f'{key}_TP'] + cmat[f'{key}_FP']) *
                        (cmat[f'{key}_TP'] + cmat[f'{key}_FN']) *
                        (cmat[f'{key}_TN'] + cmat[f'{key}_FP']) *
                        (cmat[f'{key}_TN'] + cmat[f'{key}_FN']))
        denom = check_denom(denom)
        mcc_out = numer / denom
        if np.isnan(mcc_out):
            mcc_out = -0.0
        mcc_dict[key] = mcc_out
    if as_numpy:
        return np.array(list(mcc_dict.values()), dtype=float)
    return mcc_dict


def compute_confusion_matrix(data_vec, pred_vec, dt, data_prefixes=['x', 'y'], keys=['lows', 'highs', 'spikes']):
    """
    compute_confusion_matrix

    utility function to compute confusion matrix, with rates
    """
    diffs = {}  # holds confusion matrix
    for k in keys:
        current_keys = [f'{dpre}_{k}' for dpre in data_prefixes]
        k_act = data_vec[current_keys].dropna().apply(
            lambda r: (r.array).any(), axis=1)
        k_pred = pred_vec[current_keys].apply(
            lambda r: (r.array).any(), axis=1)
        k_pred = reindex_as_data(k_pred.copy(), k_act.index.copy(), dt)
        # basic counts
        TP = np.sum(np.logical_and(k_act.array, k_pred.array))
        TN = np.sum(
            np.logical_and(np.logical_not(k_act.array),
                           np.logical_not(k_pred.array))
        )
        FP = np.sum(np.logical_and(np.logical_not(k_act.array), k_pred.array))
        FN = np.sum(np.logical_and(k_act.array, np.logical_not(k_pred.array)))
        N = TN + TP + FN + FP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # compute confusion matrix rates
            TPR = TP / check_denom(TP + FN)
            TNR = TN / check_denom(TN + FP)
            FPR = FP / check_denom(FP + TN)
            PPV = TP / check_denom(TP + FP)
            NPV = TN / check_denom(TN + FN)
            Phi = (TP + FN) / check_denom(N)  # Phi = prevalence
        key_fmt = f'{k}_' + '{}'
        diffs.update({
            key_fmt.format('TPR'): TPR,
            key_fmt.format('TNR'): TNR,
            key_fmt.format('FPR'): FPR,
            key_fmt.format('PPV'): PPV,
            key_fmt.format('NPV'): NPV,
            key_fmt.format('Phi'): Phi
        })
        # compute confusion matrix in terms of prevalence, TPR, TNR
        diffs[k + '_TP'] = N * TPR * Phi
        diffs[k + '_TN'] = N * TNR * (1.0 - Phi)
        diffs[k + '_FP'] = N * (1.0 - TNR) * (1.0 - Phi)
        diffs[k + '_FN'] = N * (1.0 - TPR) * Phi
        diffs[f'{k}_N'] = N
    return diffs


def make_time_of_day_column(df, index_col=None, tod_colname='tod'):
    dfix = df.index
    if index_col is not None:
        dfix = df[index_col]
    dfix = pd.DatetimeIndex(dfix)
    new_df = df.copy()
    new_df[tod_colname] = to_tod_hours(dfix)
    return new_df


class PFunDayNight:
    """Day/night 24-hour definitions.
    """

    #: day
    day0, day1 = 5, 20
    day_vec = np.arange(day0, day1)
    day_str = 'day'

    #: morning
    morn0, morn1 = 4, 12
    morn_vec = np.arange(morn0, morn1)
    morn_str = 'morn'

    #: afternoon
    aftn0, aftn1 = 12, 17
    aftn_vec = np.arange(aftn0, aftn1)
    aftn_str = 'aftn'

    #: night
    night0, night1 = 0, 24
    night_vec = np.append(np.arange(night0, day0), np.arange(day1, night1))
    night_str = 'night'

    #: evening
    eve0, eve1 = 17, 22
    eve_vec = np.arange(eve0, eve1)
    eve_str = 'eve'

    @staticmethod
    def get_strs():
        """Get all relevant string attributes as a list.
        """
        pfn_sorted = sorted(PFunDayNight.__dict__.items(), key=lambda d: d[0])
        return [dn_str for k, dn_str in pfn_sorted if '_str' in k and isinstance(dn_str, str)]

    @staticmethod
    def get_vecs():
        """Get all relevant vector attributes as a nested list.
        """
        pfn_sorted = sorted(PFunDayNight.__dict__.items(), key=lambda d: d[0])
        return [dn_vec for k, dn_vec in pfn_sorted if '_vec' in k and isinstance(dn_vec, np.ndarray)]

    @staticmethod
    def get_dict():
        return dict(zip(PFunDayNight.get_strs(), PFunDayNight.get_vecs()))

    @staticmethod
    def check_one(ts, vec, rtol=0.0, atol=0.5):
        return bool(np.isclose(ts, vec, atol=atol, rtol=rtol).any())

    @staticmethod
    def check_all_times(ts, as_dict=False, **kwds):
        """Determine the time bins that `ts` belongs in. 
        """
        d = PFunDayNight.get_dict()
        check_dict = {k: PFunDayNight.check_one(
            ts, vec, **kwds) for k, vec in d.items()}
        if as_dict is True:
            return check_dict
        return [k for k in check_dict if bool(check_dict[k]) is True]

    @staticmethod
    def is_day(ts, **kwds):
        return PFunDayNight.check_one(ts, PFunDayNight.day_vec, **kwds)

    @staticmethod
    def is_morn(ts, **kwds):
        return PFunDayNight.check_one(ts, PFunDayNight.morn_vec, **kwds)

    @staticmethod
    def is_aftn(ts, **kwds):
        return PFunDayNight.check_one(ts, PFunDayNight.aftn_vec, **kwds)

    @staticmethod
    def is_night(ts, **kwds):
        return PFunDayNight.check_one(ts, PFunDayNight.night_vec, **kwds)

    @staticmethod
    def is_eve(ts, **kwds):
        return PFunDayNight.check_one(ts, PFunDayNight.eve_vec, **kwds)

    @staticmethod
    def _tests():
        assert PFunDayNight.get_strs() == [
            'aftn', 'day', 'eve', 'morn', 'night']
        assert all([(a == a1).all() for a, a1 in zip(PFunDayNight.get_vecs(), [np.array([12, 13, 14, 15, 16]),
                                                                               np.array(
                                                                                   [5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                                                                               np.array(
                                                                                   [17, 18, 19, 20, 21]),
                                                                               np.array(
                                                                                   [4,  5,  6,  7,  8,  9, 10, 11]),
                                                                               np.array([0,  1,  2,  3,  4, 20, 21, 22, 23])])])
        assert PFunDayNight.check_all_times(0, as_dict=True, atol=0.5) == \
            {'aftn': False, 'day': False, 'eve': False, 'morn': False, 'night': True}
        logging.info('[passed] Success for all `PFunDayNight` tests!')


def make_extra_tags(df, index_col=None, tod_col=False):
    # add extra tags columns to df (e.g. day/night)
    ''' df: pd.DataFrame;  (dataframe to operate on)

        index_col: str; name of column to treat as reference datetime index.
            If `None` (default), `df.index` will be used.

        tod_col: str; name of column to 
    '''
    dfix = None
    # handle cases when tod_col is not specified
    if all([index_col is None, tod_col is False]):
        dfix = df.index
    elif all([index_col is not None, tod_col is False]):
        dfix = df[index_col]
    # determine day/night
    day_vec, night_vec = PFunDayNight.day_vec, PFunDayNight.night_vec
    if dfix is not None:  # tod_col is False in this case
        dfix = pd.DatetimeIndex(dfix).to_series().apply(
            lambda ts: int(ts.hour) if ts.hour < PFunDayNight.night1 else 0)
        df['tag_is_day'] = dfix.apply(lambda ts: int(ts) in day_vec)
        df['tag_is_night'] = dfix.apply(lambda ts: int(ts) in night_vec)
    elif all([dfix is None, tod_col is not False]):
        # handle cases when tod_col is specified
        try:
            df['tag_is_day'] = df[tod_col].fillna(value=-1).apply(
                lambda ts: PFunDayNight.is_day(int(ts)))
            df['tag_is_night'] = df[tod_col].fillna(value=-1).apply(
                lambda ts: PFunDayNight.is_night(int(ts)))
        except AttributeError:
            df['tag_is_day'] = int(df[tod_col]) in day_vec
            df['tag_is_night'] = int(df[tod_col]) in night_vec
    else:
        raise RuntimeError('Must specify index_col or tod_col.')
    return df


class RiskComputer(TaggedCacheMixin):
    def __init__(self, df, smin: float, smax: float, cols: Union[AnyStr, Container], ecols=[], **kwds):
        self._debug = kwds.get('debug', False)
        self._interp_kind = kwds.get("interp_kind", "none")
        self.df = df.copy()
        self.smin, self.smax = smin, smax
        self.cols = cols
        if not isinstance(self.cols, Container):
            self.cols = [self.cols, ]
        self.ecols = ecols
        self._err_vector = None
        self._nspikes = None
        self._ave_risk = None
        self._freq_lows = None
        self._freq_highs = None
        kwds.pop('ecols', None)
        self._setup(df=self.df, smin=self.smin, smax=self.smax,
                    cols=self.cols, ecols=self.ecols, **kwds)

    @property
    def debug(self):
        return self._debug

    def __call__(self, df, smin, smax, cols, **kwds):
        self._setup(df=df, smin=smin, smax=smax, cols=cols, **kwds)
        return self

    def _setup(self, df=None, smin=None, smax=None, cols=None, ecols=None, ix_col=None, reset_cache=True, **kwds):
        if reset_cache is True:
            self.invalidate_all()
        if df is None:
            df = self.df
        if smin is None:
            smin = self.smin
        if smax is None:
            smax = self.smax
        if cols is None:
            cols = self.cols
        check_names = ['df', 'smin', 'smax', 'cols']
        checks = [df is not None, smin is not None,
                  smax is not None, cols is not None]
        try:
            assert all(checks)
        except AssertionError as local_e:
            err_msg = f'\nchecks=\n{json.dumps(dict(zip(check_names, checks)))}\n'
            logging.error(err_msg)
            raise Exception(err_msg)
        self.df = df.copy().fillna(value=np.nan)
        if ix_col is None:
            ix = pd.to_datetime(self.df.index)
        else:
            ix = pd.to_datetime(self.df[ix_col])
        self.df.set_index(ix, inplace=True)
        self._interp_kind = kwds.get('interp_kind', self._interp_kind)
        # ! explicitly drop excess columns
        if any([dfc not in cols for dfc in self.df.columns]):
            try:
                self.df.drop(
                    columns=[c for c in self.df.columns if c not in cols + ecols], inplace=True)
                self.df[cols]
                if ecols:
                    self.df[ecols]
            except Exception as local_exception:
                if self.debug:
                    warn(
                        f'specified cols={cols}; actual cols={self.df.columns}.')
                raise local_exception
        self.smin, self.smax = smin, smax
        self.cols = cols
        self.ecols = ecols
        self._err_vector = None
        self._nspikes = None
        self._ave_risk = None
        self._freq_lows = None
        self._freq_highs = None
        self.identify_spikes()
        self.compute_risk(**kwds)

    @tagged_property.tag('dt_inferred')
    def dt_inferred(self):
        return to_offset(pd.infer_freq(self.df.index))

    @property
    def xcol(self):
        return self.cols[0]

    @property
    def ycol(self):
        return self.cols[-1]

    @property
    def xcol_ix(self):
        return list(self.cols).index(self.xcol)

    @property
    def ycol_ix(self):
        return list(self.cols).index(self.ycol, self.xcol_ix)

    @property
    def nspikes_observed(self):
        return self._nspikes

    @property
    def ave_risk(self):
        return self._ave_risk

    @property
    def freq_lows(self):
        return self._freq_lows

    @property
    def freq_highs(self):
        return self._freq_highs

    @tagged_property
    def risk_vector(self):
        return self.df['risk'].to_numpy(na_value=np.nan, dtype=float)

    @tagged_property
    def risk_index(self):
        return self.df['risk'].index

    @property
    def err_vector(self):
        if self._err_vector is not None:
            return self._err_vector
        if self.ecols:
            self._err_vector = self.df[self.ecols]
        else:
            tmpdf = self.df[self.cols].iloc[:, [self.xcol_ix, self.ycol_ix]]
            tmpdf.fillna(value=tmpdf.mean().mean(), inplace=True)
            tmpstd = tmpdf.rolling(window=6, center=True).std()
            tmpstd.fillna(value=tmpdf.std().mean(), inplace=True)
            self._err_vector = tmpstd / np.sqrt(float(tmpstd.shape[0]))
        return self._err_vector

    @err_vector.setter
    def err_vector(self, new_evec):
        self._err_vector = new_evec
        if self.ecols:
            self.df.drop(columns=self.ecols)
            self.ecols = None

    @tagged_property
    def lows_vector(self):
        if 'lo' in self.df.columns:
            return self.df['lo']
        return self.df[['x_lows', 'y_lows']].apply(lambda d: np.isfinite(d).all() and d.any(), axis=1)

    @tagged_property
    def highs_vector(self):
        if 'hi' in self.df.columns:
            return self.df['hi']
        return self.df[['x_highs', 'y_highs']].apply(lambda d: np.isfinite(d).all() and d.any(), axis=1)

    @tagged_property
    def spikes_vector(self):
        if 'spikes' in self.df.columns:
            return self.df['spikes']
        lohi = pd.DataFrame(
            {'lo': self.lows_vector, 'hi': self.highs_vector}, index=self.lows_vector.index)
        return lohi.apply(lambda d: d.any(), axis=1).astype(bool)

    def identify_spikes(self):
        xcol, ycol = self.xcol, self.ycol
        if any([any([dfc not in self.cols for dfc in self.df.columns]), xcol not in self.df.columns, ycol not in self.df.columns]):
            raise ValueError('(RiskComputer) DataFrame columns != [xcol, ycol],\n'
                             f'\t( [{str(self.df.columns)},] != [{xcol}, {ycol}] )')
        for j, col, char in zip([self.xcol_ix, self.ycol_ix], [xcol, ycol], ['x', 'y']):
            # ! periodic interpolation
            dfc = periodic_interp(
                self.df.iloc[:, j], self.df.index, 'auto', interp_kind=self._interp_kind)
            try:
                self.df[f'{char}_lows'] = is_low(dfc, self.smin)
                self.df[f'{char}_highs'] = is_high(dfc, self.smax)
                self.df[f'{char}_spikes'] = self.df[[f'{char}_lows', f'{char}_highs']].apply(
                    func=lambda d: np.logical_or(
                        bool(d[f'{char}_lows']), bool(d[f'{char}_highs']))
                    if not np.isnan(d.to_numpy(na_value=np.nan, dtype=float)).any() else np.nan, axis=1)
            except ValueError as local_exception:
                if int(self.debug) >= 2:
                    breakpoint()
                raise local_exception
        self._nspikes = sum_numba(self.df[['x_spikes', 'y_spikes']].fillna(
            value=False).to_numpy(dtype=float, na_value=np.nan))
        self._freq_lows = np.nanmean(
            self.df[['x_lows', 'y_lows']].to_numpy(dtype=float, na_value=np.nan))
        self._freq_highs = np.nanmean(
            self.df[['x_highs', 'y_highs']].to_numpy(dtype=float, na_value=np.nan))
        self.df = self.df.fillna(value=np.nan)

        # ! temporarily store uncorrected lows/highs
        self.df['lo'] = self.lows_vector
        self.df['hi'] = self.highs_vector

        # ! fix overlapping highs/lows
        lohi_groups = self.df.groupby(['lo', 'hi'])
        if (True, True) in lohi_groups.groups:
            overlap_df = lohi_groups.get_group((True, True))
            odf = overlap_df.copy()
            odf = odf.mean(axis=1)
            xec = odf.std()
            xptot = odf + xec
            xntot = odf - xec
            pdiff = (xptot - self.smax)
            ndiff = (self.smin - xntot)
            where_p = overlap_df.loc[pdiff < ndiff].index
            where_n = overlap_df.loc[ndiff < pdiff].index
            self.df.loc[where_p, 'lo'] = False
            self.df.loc[where_n, 'hi'] = False
            # ! update self.df with corrected lows/highs
            self.df['lo'] = self.lows_vector
            self.df['hi'] = self.highs_vector
        # set spikes vector
        self.df['spikes'] = self.spikes_vector

    @tagged_property
    def freq_spikes(self):
        return (self.freq_lows + self.freq_highs)  # type: ignore

    @tagged_property
    def freq_nans(self):
        _freq_nans = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            _freq_nans = np.nanmean(
                self.df.iloc[:, [self.xcol_ix, self.ycol_ix]].isna().mean(axis=1))
        return _freq_nans

    @tagged_property
    def na_value(self):
        """replace missing values with this in risk calculations
        """
        return float(max(self.freq_spikes, 0.5))

    @staticmethod
    def calc_uncertainty(x, xout):
        is_na = np.logical_not(np.isfinite(x))
        if is_na.any():
            risk_na = (is_na.astype(float)) * \
                np.power(np.cumsum(is_na), 3.0) / float(x.size)
            xout += risk_na
        return xout

    def calc_static_risk(self):
        """compute the static component of risk.
        """
        X = self.df[self.cols].mean(axis=1)\
            .to_numpy(dtype=float, na_value=np.nan)
        risk_vec = np.zeros((X.size), dtype=float)
        is_lo = self.df['lo'].to_numpy(dtype=float, na_value=np.nan)
        is_hi = self.df['hi'].to_numpy(dtype=float, na_value=np.nan)
        is_fine = (np.logical_not(is_lo)) & (np.logical_not(is_hi))
        if is_fine.any():
            X_min, X_max = X[is_fine].min(), X[is_fine].max()
            risk_vec[is_fine] = 0.23 * (X[is_fine] - X_min) / (X_max - X_min)
        risk_vec[np.logical_not(np.isfinite(risk_vec))] = np.nanmean(risk_vec)
        if (is_lo > 0.0).any():
            risk_vec[(is_lo > 0.0) & np.logical_not(np.isnan(is_lo))] += 1.0
        if (is_hi > 0.0).any():
            risk_vec[(is_hi > 0.0) & np.logical_not(np.isnan(is_hi))] += 1.0
        risk_vec = self.calc_uncertainty(X, risk_vec)
        return risk_vec

    def compute_risk(self, **kwds):

        # ! NaN -> np.nan (for array ops)
        self.df = self.df.fillna(value=np.nan)

        # initialize risk vector
        self.df['risk'] = np.zeros((self.df.shape[0],))

        def calc_dxdt(x, k=1, b0=self.freq_nans):
            """compute dxdt, normalize [0, 1]
            """
            dxdt = x.diff(k).abs().fillna(np.nan)
            dxdt_ave = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                dxdt_ave = np.nanmean(
                    [dxdt.abs().to_numpy(dtype=float, na_value=np.nan)])

            def _repl_missing_vals(d, b0=b0 + dxdt_ave):
                # ! handle missing values
                if d.isna().any():
                    nan_vec = 0.23 * b0 + b0 * \
                        np.linspace(0.23, 1.0, num=len(d))
                    dout = np.power(nan_vec, 3.0)
                else:
                    dout = d
                return dout
            aggfun = pd.NamedAgg(column='dx', aggfunc=lambda g: np.nanmean(
                np.array(_repl_missing_vals(g), dtype=float)))
            dftmp = pd.DataFrame(
                {'dx': dxdt, 'na': pd.isna(dxdt)}, columns=['dx', 'na'])
            # ! label consecutive NaNs
            dftmp['na_group'] = (dftmp['na'].diff(1) != 0).cumsum()
            df_repl = dftmp.groupby(
                ['na', 'na_group'], as_index=False).agg(dx_new=aggfun)
            dftmp['dx'] = dftmp.apply(lambda d: df_repl.loc[d['na_group'] == df_repl['na_group'], 'dx_new'].item()
                                      if d['na'] == 1 else d['dx'], axis=1)
            dxdt = dftmp['dx'].abs().to_numpy(dtype=float, na_value=np.nan)
            return normalize(dxdt)

        #: missing values -> spikes + missing
        na_value = self.na_value

        #: compute time-varying risk
        for j, col in zip([self.xcol_ix, self.ycol_ix], self.cols):
            if self._interp_kind != "none":
                vec_interp = periodic_interp(
                    self.df.iloc[:, j], self.df.index, 'auto', interp_kind=self._interp_kind)  # ! periodic interpolation
            else:
                # ! NOTE: if explicitly set to "none", make sure to include extra risk for missing values.
                vec_interp = self.df.iloc[:, j]
            #: ! NOTE: line below ensures that vec_interp includes NaN values in correct places.
            vec_interp = reindex_as_data(
                vec_interp.drop_duplicates(), self.df.index, self.dt_inferred)
            dxdt_normed = np.abs(calc_dxdt(vec_interp))
            dxdt_normed += np.abs(np.gradient(dxdt_normed)) / 3.0
            tv_risk = np.abs(np.log(1.0 + dxdt_normed))
            self.df['risk'] += tv_risk
        self.df['risk'] /= 2.0

        #: store as 'dynamic' risk
        self.df['dyn_risk'] = self.df.risk.to_numpy(
            dtype=float, na_value=na_value)

        #: compute stationary risk
        self.df['static_risk'] = self.calc_static_risk()
        #: include in total risk
        self.df['risk'] += self.df['static_risk']

        #: rolling average for risk
        k_win = kwds.get('k_win', kwds.get('window', kwds.get('winsize', 5)))
        rolling_opts = kwds.get('rolling_opts', {})
        window_size = k_win * pd.Timedelta(self.dt_inferred)
        self.df.set_index(pd.to_datetime(self.df.index), inplace=True)
        try:
            df_risk_rolling = self.df['risk'].rolling(
                window_size, min_periods=1, center=True, **rolling_opts)
        except (ValueError, NotImplementedError):
            df_risk_rolling = self.df['risk'].rolling(
                k_win, min_periods=1, center=True, **rolling_opts)
        self.df['risk'] = df_risk_rolling.mean().fillna(na_value)

        #: constrain risk to appropriate bounds
        self.df.loc[self.df.risk > 1.0, 'risk'] = 1.0
        self.df.loc[self.df.risk < 0.0, 'risk'] = 0.0

        # compute average risk
        self._ave_risk = np.nanmean(self.risk_vector)


class DupeColRenamer(object):
    ''' rename duplicate columns '''

    def __init__(self, df):
        self.df = df
        self.excluded = df.columns[np.logical_not(
            df.columns.duplicated(keep=False))]

    def __call__(self, newcols=None):
        inc = itertools.count().__next__
        if newcols is not None:  # ! assumes there are max of 2 cols to be renamed
            cmap = dict([(self.df.columns[i], newcols[i])
                         if newcols[i] is not None else newcols[0]
                         for i in range(len(newcols))])
            self.df = self.df.rename(columns=cmap)

        def ren(name):
            i = inc()  # starts appending after 0, see line below
            return f"{''.join(n for n in name if not n.isdigit())}{i}" if all([name not in self.excluded, i > 0]) else name
        self.df = self.df.rename(columns=ren)
        return self.df


def get_nearest_time(df, index_col, input_time, retix=False):
    ix = (pd.to_datetime(df[index_col]) -
          pd.to_datetime(input_time)).abs().argsort()[0]  # type: ignore
    xt = df.iloc[ix]
    if retix is True:
        return xt, df.index[ix]
    return xt


def fuzzy_argwhere_close(df, val, col=None, tol=0.05):
    ''' Returns indices where the dataframe (or series, if `col` is specified)
       is close (within `tol`, absolute) to the specified value (`val`).
    '''
    tdf = df
    if col is not None:
        tdf = df[col]
    ixs = np.argwhere(np.abs(tdf.to_numpy() - val) <= tol).flatten()
    return tdf.iloc[ixs].index


def inverse_sort(x_sorted, sorted_ixs):
    ''' Helper function that "re-un-sorts" a previously sorted array (x_sorted)
    [x_sorted]: sorted array that should be reverted to the provided index
    [sorted_ixs]: indices to be used for unsorting, e.g. output of `np.argsort(y)`
    '''
    x_resorted = np.array(sorted([(ix, ai)
                                  for ix, ai in zip(sorted_ixs, x_sorted)],
                                 key=lambda xx: xx[0]),
                          dtype=x_sorted.dtype).reshape(-1, 2)
    try:
        x_resorted = np.flatten(x_resorted[:, 1])  # type: ignore
    except AttributeError:
        x_resorted = x_resorted[:, 1].flatten()
    return x_resorted


def time_gradient(df: Union[pd.Series, pd.DataFrame], periods: int = 1, interp_kind: str = "none"):
    """calculate the gradient of the input Series/DataFrame as a time derivative (decimal seconds).

    [Illustrated thusly]:
        dx / dt => <d_Units> / <d_Seconds>
    """

    def calc_grad(dloc: pd.Series, copy=True):
        """NOTE: `copy=True` is recommended.

        ref: https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#mutating-with-user-defined-function-udf-methods
        """
        if copy is True:
            dloc = dloc.copy()
        dloc = periodic_interp(dloc, dloc.index, interp_kind=interp_kind)
        g_top = dloc.diff(periods=periods)
        g_bot = dloc.index.to_series().diff(periods=periods).dt.total_seconds()
        grad = (g_top / g_bot)
        grad.fillna(np.nan, inplace=True)
        grad = periodic_interp(grad, grad.index, interp_kind=interp_kind)
        return grad

    if isinstance(df, pd.Series):
        grad = calc_grad(df)
        return grad
    elif isinstance(df, pd.DataFrame):
        return df.apply(lambda dloc: calc_grad(dloc, copy=True), axis=0)
    else:
        raise TypeError(
            'Input `df` must be either a pandas Series or DataFrame.')


def auto_pct_change(df, group=None, pct=False, periods=1):
    def compute_pct_change(df):
        df = df.sort_index()
        if pct is True:
            return df.pct_change(freq=pd.infer_freq(df.index))
        return time_gradient(df, periods=periods)
    if group is None:
        return compute_pct_change(df)
    return df.groupby(level=group).apply(lambda dg: compute_pct_change(dg))
