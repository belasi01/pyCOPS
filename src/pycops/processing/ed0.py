"""Fit the above-water reference irradiance (Ed0) and derive its per-scan correction.

Port of the LOESS-fitting portion of ``process.Ed0.R``. The theoretical
clear-sky ``Ed0.th`` diagnostic (Gregg & Carder / Thuillier-based) isn't
ported -- it's a QC plot overlay, not an input to Kd/Rrs.

R's own ``Ed0.correction`` (confirmed by reading ``process.Ed0.R`` directly:
``E0d.correction <- t(aop.0 / t(aop.raw))``) always divides the LOESS-smoothed
surface reference by the *raw*, unsmoothed per-scan Ed0 -- so the correction
inherits Ed0's own scan-to-scan sensor noise, which then propagates into every
instrument it's multiplied into. ``correction_smoothed`` below is a pycops-only
alternative (no R equivalent) that divides by the LOESS-smoothed Ed0 value at
each scan's own elapsed time instead, for a researcher who finds the raw-based
correction too noisy (see ``fit_ed0_for_cast``/``process_cast`` for how the
choice between the two is resolved).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pycops.processing.loess import loess_1d
from pycops.processing.profile_fit import fit_profile_loess

Ed0CorrectionMethod = Literal["raw", "smoothed"]


@dataclass(frozen=True)
class Ed0Fit:
    """Result of fitting Ed0 and deriving its per-scan illumination correction."""

    fitted: np.ndarray  # (n_depth_grid, n_waves)
    value_at_0: np.ndarray  # (n_waves,) -- Ed0.0p, the smoothed surface reference
    correction_raw: np.ndarray  # (n_scans, n_waves) -- Ed0.0p / each raw scan
    correction_smoothed: np.ndarray  # (n_scans, n_waves) -- Ed0.0p / the LOESS-smoothed Ed0 at that scan's own elapsed time
    method: Ed0CorrectionMethod  # which of the above is "correction" below
    correction: np.ndarray  # (n_scans, n_waves) -- correction_raw or correction_smoothed, per method


def fit_ed0(
    waves: np.ndarray,
    depth_kept: np.ndarray,
    ed0_kept: np.ndarray,
    ed0_all: np.ndarray,
    span: float,
    depth_grid: np.ndarray,
    idx_depth_0: int = 0,
    time_kept: np.ndarray | None = None,
    time_all: np.ndarray | None = None,
    method: Ed0CorrectionMethod = "raw",
) -> Ed0Fit:
    """LOESS-fit Ed0 vs depth (a monotonic proxy for time) and derive its correction.

    ``depth_kept``/``ed0_kept`` are the tilt- and depth-QC'd subset used for
    the fit itself; ``ed0_all`` is the *full*, unfiltered raw Ed0 matrix --
    the resulting correction is the ratio of the smoothed surface reference
    to every individual raw scan (including ones later dropped by QC), since
    it's applied to EdZ/LuZ/EuZ before their own filtering, matching the R
    package's processing order.

    ``time_kept``/``time_all`` (elapsed seconds, same subset/order convention
    as ``depth_kept``/``ed0_all``; defaulting to ``depth_kept``/``depth_kept``
    when omitted, e.g. by callers that only ever want the raw correction) are
    used for ``correction_smoothed``, via a *separate* LOESS fit directly
    against elapsed time rather than depth.

    **Real-data bug found and fixed (2026-08-08)**: an earlier version of this
    function evaluated the depth-domain LOESS fit (the same one used for
    ``fitted``/``value_at_0``) across the whole profile and interpolated it
    back onto each scan's own *depth* to build ``correction_smoothed``. Depth
    is only ever "a monotonic proxy for time" *when the cast is monotonic* --
    real casts routinely aren't: a pre-descent bobbing phase at the surface
    (several scans oscillating within a few cm of each other, confirmed on a
    real CASCADE cast) and any down-up profile (confirmed on two more real
    CASCADE casts: depth increases to a maximum then decreases again on the
    way back up) both put *two different real times* at the *same* depth.
    Interpolating a depth-indexed curve back onto such a profile blends
    together Ed0 readings from genuinely different moments, producing a
    ``correction_smoothed`` that's locally noisy right where the cast is
    non-monotonic (the bobbing start) and diverges from the raw signal
    wherever the up-leg's depths retrace the down-leg's (the end of a
    down-up cast) -- exactly what Simon reported after reprocessing real
    CASCADE data. Elapsed time has no such ambiguity (each scan has its own
    distinct time by construction), so ``correction_smoothed`` is now a
    genuinely separate LOESS fit/evaluation against ``time_kept``/``time_all``
    (:func:`pycops.processing.loess.loess_1d` directly, not
    :func:`pycops.processing.profile_fit.fit_profile_loess`'s depth-grid-
    indexed machinery, which assumes a monotonic, sorted depth axis that
    elapsed time doesn't need). Reuses the exact same span *fraction*
    ``fit_profile_loess`` would have derived for the depth-domain fit
    (``min(1.0, span / ptp(depth_kept))``) -- valid because
    ``loess_1d``'s span is already a fraction of the data's point count, not
    a physical bandwidth, so it carries over unchanged from one x-axis to
    another.

    ``method`` picks which of ``correction_raw``/``correction_smoothed`` is
    returned as the plain ``correction`` field (what ``fit_cast`` actually
    multiplies into EdZ/LuZ/EuZ) -- both are always computed and returned, so
    a caller (e.g. the diagnostic "Ed0 stability" plot) can compare them
    regardless of which one is active.
    """
    time_kept = depth_kept if time_kept is None else np.asarray(time_kept, dtype=float)
    time_all = depth_kept if time_all is None else np.asarray(time_all, dtype=float)
    ed0_all = np.asarray(ed0_all, dtype=float)

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
    correction_raw = profile.value_at_0[None, :] / ed0_all

    depth_range = np.ptp(depth_kept)
    time_span_fraction = min(1.0, span / depth_range) if depth_range > 0 else 1.0
    # Clamp query times to the *training* data's own range rather than let the degree-2 fit
    # extrapolate freely -- same "clamp, don't extrapolate" convention aop_cleaning.py already
    # uses for its own spline. Real-data bug found on a real CASCADE cast with a saved
    # time.window: scans outside the kept training window (e.g. the pre-descent bobbing phase
    # before the window starts) were queried tens of seconds beyond the fit's own trained range,
    # and a degree-2 polynomial extrapolated that far can wildly overshoot -- confirmed on real
    # data as negative "corrected" Ed0 (i.e. correction_smoothed flipping sign) right at the start
    # of the cast. Clamping keeps every query point a genuine interpolation.
    time_query = np.clip(time_all, time_kept.min(), time_kept.max())
    ed0_smoothed_at_scan = np.column_stack(
        [loess_1d(time_kept, ed0_kept[:, i], time_query, span=time_span_fraction) for i in range(len(waves))]
    )
    correction_smoothed = profile.value_at_0[None, :] / ed0_smoothed_at_scan

    correction = correction_smoothed if method == "smoothed" else correction_raw
    return Ed0Fit(
        fitted=profile.fitted,
        value_at_0=profile.value_at_0,
        correction_raw=correction_raw,
        correction_smoothed=correction_smoothed,
        method=method,
        correction=correction,
    )
