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


def kd_at_light_fraction(
    fitted_profile: np.ndarray,
    depth_grid: np.ndarray,
    ed0_subsurface: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Mean diffuse attenuation from the surface down to the depth where light has attenuated
    to ``fraction`` of its subsurface value (e.g. 0.01/0.1 for the 1%/10% light levels, or
    ``1/e`` for the penetration depth), one value per wavelength.

    Port of ``generate.cops.DB.R``'s inline spline-based depth search (``z1``/``z10``/``zpd``):
    for each wavelength, the fraction of subsurface light remaining at each grid depth
    (``fitted_profile / ed0_subsurface``, NaN treated as 0 -- matches R's
    ``percentEdZ[is.na(percentEdZ)] <- 0``) is inverted to find the depth ``z`` where it equals
    ``fraction``, then ``Kd = -ln(fraction) / z``. Uses ``np.interp`` rather than R's ``spline()``
    since only a single crossing depth is needed, not a smooth curve -- matching this module's own
    depth-axis convention (:func:`compute_K` also works directly off ``depth_grid``, no spline).

    Unlike R's ``spline()``, ``np.interp`` never extrapolates -- it clips to the nearest measured
    depth instead of continuing past it. So when ``fraction`` falls outside the profile's own
    observed range (light never actually attenuated that far within the measured depths), this
    explicitly returns ``NaN`` rather than silently reusing the deepest measured point as if it
    were the true crossing depth. R instead lets ``spline()`` extrapolate and then rejects the
    result via ``z1[z1<0] <- NA; z1[z1 > max(depth.fitted)] <- NA`` -- which *usually* catches an
    unreachable crossing (an unconstrained extrapolation tends to overshoot well past
    ``max(depth.fitted)``) but isn't guaranteed to for every profile shape; refusing outright is
    the more conservative, dependency-free choice here.
    """
    fitted_profile = np.asarray(fitted_profile, dtype=float)
    depth_grid = np.asarray(depth_grid, dtype=float)
    ed0_subsurface = np.asarray(ed0_subsurface, dtype=float)

    percent = fitted_profile / ed0_subsurface[None, :]
    percent = np.nan_to_num(percent, nan=0.0)

    n_waves = fitted_profile.shape[1]
    z = np.full(n_waves, np.nan)
    for i in range(n_waves):
        # np.interp needs its x-coordinates ascending; percent decreases with depth, so reverse
        # both arrays (depth_grid is itself already ascending).
        xp = percent[::-1, i]
        if fraction < xp.min() or fraction > xp.max():
            continue  # outside the profile's own observed range -- leave NaN, see docstring
        z[i] = np.interp(fraction, xp, depth_grid[::-1])
    z[(z < 0) | (z > depth_grid.max())] = np.nan

    return -np.log(fraction) / z
