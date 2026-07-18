"""Diffuse attenuation coefficient from a LOESS-fitted depth profile.

Port of ``compute.K.R``.
"""

from __future__ import annotations

import numpy as np


def compute_K(
    depth_grid: np.ndarray,
    idx_depth_0: int,
    value_at_0: np.ndarray,
    fitted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Local (``KZ``) and depth-integrated (``K0``) attenuation coefficients.

    ``fitted`` is ``(n_depth_grid, n_waves)`` (as returned by
    :func:`pycops.processing.profile_fit.fit_profile_loess`); both outputs
    are ``(n_depth_grid - 1, n_waves)``, aligned with ``depth_grid[1:]``.

    ``KZ`` is the pointwise derivative of ``-d(ln aop)/dz`` between
    consecutive grid points. ``K0`` is the average attenuation from the
    surface reference depth (``depth_grid[idx_depth_0]``) down to each grid
    point.
    """
    depth_grid = np.asarray(depth_grid, dtype=float)
    value_at_0 = np.asarray(value_at_0, dtype=float)
    fitted = np.asarray(fitted, dtype=float)

    log_fitted = np.log(fitted)
    KZ = -np.diff(log_fitted, axis=0) / np.diff(depth_grid)[:, None]

    z0 = depth_grid[idx_depth_0]
    K0 = (np.log(value_at_0)[None, :] - log_fitted[1:, :]) / (depth_grid[1:] - z0)[:, None]

    return KZ, K0
