"""Local polynomial regression (LOESS), matching R's ``stats::loess`` with
``family="gaussian"`` and ``control=loess.control(surface="direct")``.

R's implementation (used throughout the ``Cops`` package's depth fitting) is a
degree-2 local weighted least-squares regression with a tricube kernel, span
expressed as a *fraction of the data* rather than a bandwidth in data units,
and ``surface="direct"`` meaning each requested point is fit exactly rather
than interpolated from a coarser grid -- important since depth-profile fits
are often evaluated at or beyond the shallowest measured depth. There is no
maintained Python binding of the original Fortran routine, so this ports the
algorithm directly rather than adding a compiled dependency.
"""

from __future__ import annotations

import numpy as np


def loess_1d(x: np.ndarray, y: np.ndarray, xout: np.ndarray, span: float, degree: int = 2) -> np.ndarray:
    """Fit a local weighted polynomial regression of ``y`` on ``x`` and evaluate at ``xout``.

    ``span`` is the fraction of points (0 < span <= 1, clipped to at least
    enough points to fit the polynomial) used in each local neighborhood.
    Returns NaN at any ``xout`` point too far from data to fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xout = np.asarray(xout, dtype=float)

    n = len(x)
    k = int(np.ceil(span * n))
    k = min(max(k, degree + 1), n)

    out = np.full(xout.shape, np.nan)
    for j, x0 in enumerate(xout):
        d = np.abs(x - x0)
        neighbors = np.argpartition(d, k - 1)[:k]
        dmax = d[neighbors].max()
        if dmax == 0:
            out[j] = y[neighbors].mean()
            continue

        u = d[neighbors] / dmax
        w = np.where(u < 1, (1 - u**3) ** 3, 0.0)
        sqrt_w = np.sqrt(w)

        dx = x[neighbors] - x0
        design = np.vander(dx, degree + 1, increasing=True)
        try:
            beta, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y[neighbors] * sqrt_w, rcond=None)
            out[j] = beta[0]
        except np.linalg.LinAlgError:
            out[j] = np.nan

    return out
