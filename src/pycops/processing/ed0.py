"""Fit the above-water reference irradiance (Ed0) and derive its per-scan correction.

Port of the LOESS-fitting portion of ``process.Ed0.R``. The theoretical
clear-sky ``Ed0.th`` diagnostic (Gregg & Carder / Thuillier-based) isn't
ported -- it's a QC plot overlay, not an input to Kd/Rrs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycops.processing.profile_fit import fit_profile_loess


@dataclass(frozen=True)
class Ed0Fit:
    """Result of fitting Ed0 and deriving its per-scan illumination correction."""

    fitted: np.ndarray  # (n_depth_grid, n_waves)
    value_at_0: np.ndarray  # (n_waves,) -- Ed0.0p, the smoothed surface reference
    correction: np.ndarray  # (n_scans, n_waves) -- Ed0.0p / each raw scan


def fit_ed0(
    waves: np.ndarray,
    depth_kept: np.ndarray,
    ed0_kept: np.ndarray,
    ed0_all: np.ndarray,
    span: float,
    depth_grid: np.ndarray,
    idx_depth_0: int = 0,
) -> Ed0Fit:
    """LOESS-fit Ed0 vs depth (a monotonic proxy for time) and derive its correction.

    ``depth_kept``/``ed0_kept`` are the tilt- and depth-QC'd subset used for
    the fit itself; ``ed0_all`` is the *full*, unfiltered raw Ed0 matrix --
    the resulting correction is the ratio of the smoothed surface reference
    to every individual raw scan (including ones later dropped by QC), since
    it's applied to EdZ/LuZ/EuZ before their own filtering, matching the R
    package's processing order.
    """
    profile = fit_profile_loess(
        waves,
        depth_kept,
        ed0_kept,
        span,
        depth_grid,
        idx_depth_0=idx_depth_0,
        span_wave_correction=False,
        depth_span=True,
    )
    correction = profile.value_at_0[None, :] / np.asarray(ed0_all, dtype=float)
    return Ed0Fit(fitted=profile.fitted, value_at_0=profile.value_at_0, correction=correction)
