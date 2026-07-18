"""Extrapolate a radiometric profile to the surface via log-linear regression.

Port of ``compute.Ksurf.linear.R``. For each wavelength, searches over a
range of near-surface layer thicknesses (0.1 m steps, from 0.30 m to
``delta_depth`` below the shallowest kept measurement) for the thickest layer
whose log-linear fit is both well-behaved (its depths pass a Kolmogorov-Smirnov
test against an evenly spaced reference, guarding against a bimodal/clumped
sample) and has the best R² among those, then keeps it if that R² clears
``r2_threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SurfaceLinearFit:
    """Per-wavelength result of the near-surface log-linear extrapolation."""

    value_at_surface: np.ndarray  # X.0m: the fitted quantity just below the surface
    k_surf: np.ndarray  # Kx: the surface-layer attenuation coefficient
    z_interval: np.ndarray  # depth of the bottom of the layer used
    ix_z_interval: np.ndarray  # index into `depth` of that bottom
    r2: np.ndarray
    ks_pvalue: np.ndarray


def fit_surface_linear(
    depth: np.ndarray,
    aop: np.ndarray,
    detection_limit: np.ndarray,
    r2_threshold: float = 0.80,
    delta_depth: float = 2.5,
) -> SurfaceLinearFit:
    """Fit ``aop`` (n_scans, n_waves), assumed time/depth-ordered, near the surface.

    ``detection_limit`` is a per-wavelength array; scans at or below it are
    excluded from each candidate fit.
    """
    depth = np.asarray(depth, dtype=float)
    aop = np.asarray(aop, dtype=float)
    detection_limit = np.asarray(detection_limit, dtype=float)
    n_waves = aop.shape[1]

    value_at_surface = np.full(n_waves, np.nan)
    k_surf = np.full(n_waves, np.nan)
    z_interval = np.full(n_waves, np.nan)
    ix_z_interval = np.full(n_waves, -1, dtype=int)
    r2_out = np.full(n_waves, np.nan)
    ks_pvalue_out = np.full(n_waves, np.nan)

    min_depth_overall = np.min(depth)
    max_depth = delta_depth + min_depth_overall
    min_depth = min_depth_overall + 0.30
    n_steps = int(round((max_depth - min_depth) / 0.1)) + 1
    depth_intervals = min_depth + 0.1 * np.arange(max(n_steps, 0))

    ix_max_depth = int(np.argmin(np.abs(depth - max_depth)))

    for w in range(n_waves):
        col = aop[:, w]
        if np.sum(col[: ix_max_depth + 1] > detection_limit[w]) <= 10:
            continue

        r2 = np.full(len(depth_intervals), np.nan)
        ks_p = np.full(len(depth_intervals), np.nan)
        x0_tmp = np.full(len(depth_intervals), np.nan)
        kx_tmp = np.full(len(depth_intervals), np.nan)
        ix_z = np.full(len(depth_intervals), -1, dtype=int)

        for i, target in enumerate(depth_intervals):
            iz = int(np.argmin(np.abs(depth - target)))
            ix_z[i] = iz
            idx = np.where(col[: iz + 1] > detection_limit[w])[0]
            n_good = len(idx)
            if n_good <= 10:
                continue

            z = depth[idx]
            log_e = np.log(col[idx])
            slope, intercept = np.polyfit(z, log_e, 1)
            resid = log_e - (intercept + slope * z)
            sse = np.sum(resid**2)
            sst = np.sum((log_e - log_e.mean()) ** 2)
            if sst <= 0:
                continue
            r_squared = 1 - sse / sst
            r2[i] = 1 - (1 - r_squared) * (n_good - 1) / (n_good - 2)
            x0_tmp[i] = np.exp(intercept)
            kx_tmp[i] = -slope

            expected = np.linspace(z[0], z[-1], n_good)
            ks_p[i] = stats.ks_2samp(z, expected).pvalue

        valid = np.where((ks_p > 0.10) & np.isfinite(r2))[0]
        if len(valid) > 1:
            best = valid[np.argmax(r2[valid])]
            if r2[best] > r2_threshold:
                value_at_surface[w] = x0_tmp[best]
                k_surf[w] = kx_tmp[best]
                z_interval[w] = depth_intervals[best]
                ix_z_interval[w] = ix_z[best]
                r2_out[w] = r2[best]
                ks_pvalue_out[w] = ks_p[best]

    return SurfaceLinearFit(value_at_surface, k_surf, z_interval, ix_z_interval, r2_out, ks_pvalue_out)
