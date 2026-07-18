"""Fit a spectral depth profile onto the depth grid, wavelength by wavelength.

Port of ``fit.with.loess`` from ``fit.functions.R``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycops.processing.loess import loess_1d


@dataclass(frozen=True)
class ProfileLoessFit:
    """Result of fitting one instrument's spectral profile onto the depth grid."""

    fitted: np.ndarray  # (n_depth_grid, n_waves), NaN where unfitted
    value_at_0: np.ndarray  # (n_waves,) -- the fit evaluated at depth_grid[idx_depth_0]


def fit_profile_loess(
    waves: np.ndarray,
    depth: np.ndarray,
    aop: np.ndarray,
    span: float,
    depth_grid: np.ndarray,
    idx_depth_0: int = 0,
    span_wave_correction: bool = False,
    depth_span: bool = True,
    minimum_obs: int = 3,
) -> ProfileLoessFit:
    """LOESS-fit each wavelength column of ``aop`` (n_scans, n_waves) against ``depth``.

    Only ``depth_grid[idx_depth_0:]`` is fitted (profiles aren't extrapolated
    to depths shallower than the reference surface point). ``span_wave_correction``
    narrows the span at short wavelengths (steeper attenuation needs finer
    smoothing), matching the R package's empirical correction. If
    ``depth_span``, ``span`` is a *depth* interval converted to a fraction of
    the data's depth range; otherwise it is already a fraction in [0, 1].
    Wavelengths with fewer than ``minimum_obs`` finite observations are left
    as NaN.
    """
    waves = np.asarray(waves, dtype=float)
    depth = np.asarray(depth, dtype=float)
    aop = np.asarray(aop, dtype=float)
    depth_grid = np.asarray(depth_grid, dtype=float)

    n_waves = aop.shape[1]
    n_grid = len(depth_grid)
    fitted = np.full((n_grid, n_waves), np.nan)
    value_at_0 = np.full(n_waves, np.nan)

    fit_slice = slice(idx_depth_0, n_grid)
    fit_targets = depth_grid[fit_slice]

    for i in range(n_waves):
        if span_wave_correction:
            span_w_corr = min(650.0 / waves[i], 1.0)
            if waves[i] < 420:
                span_w_corr = waves[i] / 420.0
        else:
            span_w_corr = 1.0

        col = aop[:, i]
        good = np.isfinite(col)
        if good.sum() <= minimum_obs:
            continue

        if depth_span:
            depth_range = np.ptp(depth[good])
            actual_span = min(1.0, (span * span_w_corr) / depth_range) if depth_range > 0 else 1.0
        else:
            actual_span = span * span_w_corr

        fitted[fit_slice, i] = loess_1d(depth[good], col[good], fit_targets, span=actual_span)
        value_at_0[i] = loess_1d(depth[good], col[good], depth_grid[idx_depth_0 : idx_depth_0 + 1], span=actual_span)[0]

    value_at_0[value_at_0 == 0] = np.nan
    return ProfileLoessFit(fitted=fitted, value_at_0=value_at_0)
