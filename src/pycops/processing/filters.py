"""Generic 1-D median-filter QC, ported from ``filtre.mediane.R``."""

from __future__ import annotations

import numpy as np


def median_filter(
    y: np.ndarray,
    k: int,
    delta: float = 0.0,
    fill: bool = True,
    replace: bool = True,
) -> np.ndarray:
    """Flag or replace outliers in ``y`` using a centered median filter.

    For each point with a full ``2k + 1``-wide window, if it differs from the
    window's median by more than ``delta`` it is either replaced by that
    median (``replace=True``) or set to NaN (``replace=False``). Points too
    close to either edge to have a full window are left untouched if
    ``fill=True``, or set to NaN if ``fill=False``.
    """
    y = np.asarray(y, dtype=float)
    f = y.copy()
    f[np.isinf(f)] = np.nan

    n = len(y)
    if n < 2 * k + 2:
        return f

    for i in range(k, n - k):
        window = y[i - k : i + k + 1]
        if not np.isnan(y[i]) and not np.all(np.isnan(window)):
            w_median = np.nanmedian(window)
            if abs(y[i] - w_median) > delta:
                f[i] = w_median if replace else np.nan

    if not fill:
        f[:k] = np.nan
        f[n - k :] = np.nan

    return f
