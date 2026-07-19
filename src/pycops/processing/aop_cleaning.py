"""Secondary, spline-based detection-limit cleaning of a fitted depth profile.

Port of the "Clean calculated AOP" block shared (identically, module a
per-instrument detection-limit table) by ``process.EdZ.R``, ``process.LuZ.R``,
and ``process.EuZ.R``: after the primary LOESS depth-profile fit
(:func:`pycops.processing.profile_fit.fit_profile_loess`), an *independent*
smoothing-spline pass over the broader tilt/depth-QC'd raw scans (before the
sub-surface-removed-layer filter, i.e. more scans than the primary fit saw)
flags additional near-detection-limit points the primary fit's own masking
missed, and NaNs them out of the fitted profile, ``KZ``, ``K0``, and the
surface value. Simon (the R package's author) added this specifically to
avoid fitting noise that the detection limit is meant to catch.

Deliberately deviates from R in one respect: R fixes ``smooth.spline``'s
smoothing strength at a hand-tuned constant (``spar=0.2``, per the R source's
own comment "spar was added to make the data smoother"). There is no
principled way to translate one specific ``spar`` value into
``scipy.interpolate.make_smoothing_spline``'s ``lam`` -- R's spar<->lambda
mapping depends on the B-spline basis's eigenvalue ratio for that exact
dataset (internal to R's compiled GCVSPL-derived code), so any fixed ``lam``
chosen to match one synthetic test case would not generalize across casts
with different scan counts/depth ranges. Uses ``make_smoothing_spline``'s own
GCV-selected ``lam`` instead: same statistical foundation (Green & Silverman
1994 penalized regression splines), but data-adaptive rather than a constant
tuned for one dataset. Checked against real R output (``smooth.spline(depth,
noisy, spar=0.2)`` on a synthetic noisy exponential-decay profile): GCV
tracks R's fit within a few percent away from the depth boundaries, which is
enough for this step's actual purpose -- thresholding against a detection
limit, not reproducing the curve's exact values.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_smoothing_spline

_MINIMUM_OBS = 10


def secondary_clean(
    depth_qc: np.ndarray,
    aop_qc: np.ndarray,
    depth_grid: np.ndarray,
    idx_depth_0: int,
    detection_limit: np.ndarray,
    depth_first_kept: float,
    aop_fitted: np.ndarray,
    value_at_0: np.ndarray,
    KZ: np.ndarray,
    K0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flag additional near-detection-limit points via a secondary spline fit.

    ``depth_qc``/``aop_qc`` are the tilt+depth-QC'd (but not sub-surface-
    removed-layer-filtered) scans for this instrument -- ``(n_scans,)`` and
    ``(n_scans, n_waves)`` respectively (R's ``Depth.all``/``aop.all``).
    ``depth_first_kept`` is the shallowest depth in the *primary* fit's kept
    set (R's ``Depth[1]``, used as a masking guard). ``aop_fitted``,
    ``value_at_0``, ``KZ``, ``K0`` are the primary LOESS fit's outputs (see
    :func:`pycops.processing.attenuation.compute_K`); returns cleaned copies
    of all four, matching each array's original shape.
    """
    depth_qc = np.asarray(depth_qc, dtype=float)
    aop_qc = np.clip(np.asarray(aop_qc, dtype=float), 0.0, None)
    depth_grid = np.asarray(depth_grid, dtype=float)

    aop_fitted = aop_fitted.copy()
    value_at_0 = value_at_0.copy()
    KZ = KZ.copy()
    K0 = K0.copy()

    n_waves = aop_fitted.shape[1]
    for w in range(n_waves):
        if np.all(np.isnan(aop_fitted[:, w])):
            continue

        lim = detection_limit[w]
        primary_below = aop_fitted[:, w] <= lim

        aop_spline = _fit_spline(depth_qc, aop_qc[:, w], depth_grid)
        if aop_spline is None:
            # Too few QC'd scans for a spline fit (an edge case unlikely on
            # real casts): fall back to the primary detection-limit check.
            combined_mask = primary_below
        else:
            secondary_below = (aop_spline <= lim) & (depth_grid > depth_first_kept)
            combined_mask = primary_below | secondary_below

        aop_fitted[combined_mask, w] = np.nan
        KZ[combined_mask[1:], w] = np.nan
        K0[combined_mask[1:], w] = np.nan

    value_at_0[:] = aop_fitted[idx_depth_0, :]

    return aop_fitted, value_at_0, KZ, K0


def _fit_spline(depth_qc: np.ndarray, aop_qc_w: np.ndarray, depth_grid: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(depth_qc) & np.isfinite(aop_qc_w)
    if valid.sum() < _MINIMUM_OBS:
        return None

    order = np.argsort(depth_qc[valid])
    x = depth_qc[valid][order]
    y = aop_qc_w[valid][order]
    x, unique_idx = np.unique(x, return_index=True)
    if len(x) < _MINIMUM_OBS:
        return None
    y = y[unique_idx]

    spline = make_smoothing_spline(x, y)
    # Clamp to the fitted range rather than extrapolating: a cubic B-spline
    # can blow up outside its knots, and depth_grid can extend shallower
    # than depth_qc's minimum (the QC'd raw scans start below the surface).
    within_range = np.clip(depth_grid, x[0], x[-1])
    return spline(within_range)
