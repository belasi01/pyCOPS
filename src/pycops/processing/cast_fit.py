"""Wire the tilt/depth QC, Ed0 correction, and LOESS/linear surface fitting
together for one cast, reading their parameters from ``init.cops.dat``.

This is the glue that was previously hand-assembled ad hoc (see the
validation notes in ``CLAUDE.md``): given an ``xarray.Dataset`` from
:func:`pycops.io.raw.read_cast` and the parsed ``init.cops.dat`` dict from
:func:`pycops.io.config.read_init_cops`, :func:`fit_ed0_for_cast` and
:func:`fit_cast` reproduce the relevant parts of ``process.Ed0.R`` /
``process.EdZ.R`` / ``process.LuZ.R`` / ``process.EuZ.R``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from pycops.processing.aop_cleaning import secondary_clean
from pycops.processing.attenuation import compute_K
from pycops.processing.depth import depth_grid as build_depth_grid
from pycops.processing.depth import good_depth_mask
from pycops.processing.detection_limits import detection_limit_for_waves
from pycops.processing.ed0 import Ed0Fit, fit_ed0
from pycops.processing.profile_fit import fit_profile_loess
from pycops.processing.surface_linear import SurfaceLinearFit, fit_surface_linear
from pycops.processing.tilt import tilt_mask

_DEPTH_PROFILED_INSTRUMENTS = ("EdZ", "LuZ", "EuZ")


def fit_ed0_for_cast(ds: xr.Dataset, init: dict[str, object]) -> Ed0Fit:
    """Fit the above-water reference (Ed0) for a cast, per ``process.Ed0.R``.

    Its ``correction`` is the illumination-adjustment factor to apply to
    EdZ/LuZ/EuZ (see :func:`fit_cast`) before their own QC and fitting.
    """
    waves = ds["wavelength"].values
    depth_ref = ds[f"{init['depth.is.on']}_Depth"].values
    depth_good = good_depth_mask(depth_ref)
    ed0_tilt_ok = tilt_mask(ds, "Ed0", init["tiltmax.optics"]["Ed0"]).values
    kept = depth_good & ed0_tilt_ok

    ed0_all = ds["Ed0"].values
    return fit_ed0(
        waves,
        depth_ref[kept],
        ed0_all[kept],
        ed0_all,
        span=init["depth.interval.for.smoothing.optics"]["Ed0"],
        depth_grid=np.array([0.0]),
        idx_depth_0=0,
    )


@dataclass(frozen=True)
class InstrumentFit:
    """Result of fitting one depth-profiled instrument (EdZ/LuZ/EuZ) for a cast."""

    instrument: str
    kept: np.ndarray  # boolean mask into the cast's scans
    depth_grid: np.ndarray
    idx_depth_0: int
    detection_limit: np.ndarray
    aop_fitted: np.ndarray  # LOESS fit onto depth_grid, real (non-log) space
    value_at_0: np.ndarray  # LOESS fit at depth_grid[idx_depth_0]
    KZ: np.ndarray  # local attenuation coefficient
    K0: np.ndarray  # depth-integrated attenuation coefficient
    surface_linear: SurfaceLinearFit


def fit_cast(ds: xr.Dataset, init: dict[str, object], instrument: str, ed0_fit: Ed0Fit) -> InstrumentFit:
    """Fit ``instrument`` (``"EdZ"``, ``"LuZ"``, or ``"EuZ"``) for one cast.

    Applies Ed0's illumination correction, tilt and depth QC, and the
    sub-surface-removed layer, then fits both the LOESS depth profile (with
    detection-limit masking, matching ``process.EdZ.R``/``process.LuZ.R``/
    ``process.EuZ.R``) and the near-surface log-linear extrapolation.
    ``ed0_fit`` comes from :func:`fit_ed0_for_cast` (shared across
    instruments, computed once per cast).
    """
    if instrument not in _DEPTH_PROFILED_INSTRUMENTS:
        raise ValueError(f"instrument must be one of {_DEPTH_PROFILED_INSTRUMENTS}, got {instrument!r}")

    waves = ds["wavelength"].values
    depth_ref = ds[f"{init['depth.is.on']}_Depth"].values
    depth_good = good_depth_mask(depth_ref)

    depth = depth_ref + init["delta.capteur.optics"][instrument]
    tilt_ok = tilt_mask(ds, instrument, init["tiltmax.optics"][instrument]).values
    depth_tilt_ok = depth_good & tilt_ok
    kept = depth_tilt_ok & (depth > init["sub.surface.removed.layer.optics"][instrument])

    aop = ds[instrument].values * ed0_fit.correction
    detection_limit = detection_limit_for_waves(instrument, waves)

    grid = build_depth_grid(init["depth.discretization"], max_depth=float(depth[kept].max()))
    idx_depth_0 = int(np.argmin(np.abs(grid - 0.0)))

    masked = aop[kept].copy()
    below_limit = masked <= detection_limit[None, :]
    masked[below_limit] = np.nan

    loess = fit_profile_loess(
        waves,
        depth[kept],
        np.log(masked),
        span=init["depth.interval.for.smoothing.optics"][instrument],
        depth_grid=grid,
        idx_depth_0=idx_depth_0,
        span_wave_correction=True,
        depth_span=True,
        minimum_obs=10,
    )
    aop_fitted = np.exp(loess.fitted)
    value_at_0 = np.exp(loess.value_at_0)
    KZ, K0 = compute_K(grid, idx_depth_0, value_at_0, aop_fitted)

    aop_fitted, value_at_0, KZ, K0 = secondary_clean(
        depth[depth_tilt_ok],
        aop[depth_tilt_ok],
        grid,
        idx_depth_0,
        detection_limit,
        depth_first_kept=float(depth[kept][0]),
        aop_fitted=aop_fitted,
        value_at_0=value_at_0,
        KZ=KZ,
        K0=K0,
    )

    linear = fit_surface_linear(
        depth[kept],
        aop[kept],
        detection_limit,
        r2_threshold=init["linear.fit.Rsquared.threshold.optics"][instrument],
        delta_depth=init["linear.fit.max.delta.depth.optics"][instrument],
    )

    return InstrumentFit(
        instrument=instrument,
        kept=kept,
        depth_grid=grid,
        idx_depth_0=idx_depth_0,
        detection_limit=detection_limit,
        aop_fitted=aop_fitted,
        value_at_0=value_at_0,
        KZ=KZ,
        K0=K0,
        surface_linear=linear,
    )
