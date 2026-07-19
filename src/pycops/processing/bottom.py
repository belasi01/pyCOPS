"""Bottom (substrate) reflectance for shallow-water ("SHALLOW") casts.

Port of ``compute.bottom.R``: when a cast's profile ends right at (or just
above) the bottom -- flagged per-cast by ``select.cops.dat``'s 4th field
(``"1"`` -- see :attr:`pycops.io.discovery.CastSelection.shallow`) --
LOESS-extrapolates the LuZ/EuZ-to-EdZ upwelling reflectance ratio down to an
estimated bottom depth, giving a rough substrate reflectance spectrum
(``Rb``) in addition to the usual open-water Rrs.

``derived.data.R`` builds one shared depth grid for a whole cast, used by
every instrument; pycops builds one *per instrument* (see ``cast_fit.py``)
since each instrument's own kept-scan max depth can differ. This module
interpolates EdZ's fitted profile onto the numerator instrument's (LuZ's/
EuZ's) own depth grid before dividing, rather than requiring them to already
match exactly -- the R source doesn't need this since it never hits the
problem in the first place.

The "both LuZ and EuZ present" bottom Q-factor (``Rb.Q`` in the R source,
``EuZ.fitted/LuZ.fitted`` extrapolated the same way) isn't ported here --
only the two primary substrate-reflectance outputs are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycops.processing.cast_fit import InstrumentFit
from pycops.processing.profile_fit import fit_profile_loess

_BOTTOM_INSTRUMENTS = ("LuZ", "EuZ")
_PI_FACTOR = {"LuZ": np.pi, "EuZ": 1.0}  # LuZ is a radiance -> needs pi to become an irradiance ratio
_NEAR_BOTTOM_MARGIN_M = 0.30


def compute_bottom_depth(
    depth_is_on_depth: np.ndarray,
    kept: np.ndarray,
    delta_capteur_m: float,
    distance_above_bottom_m: float = 0.15,
) -> float:
    """Estimated bottom depth for a shallow cast.

    The deepest kept scan of the reference (``depth.is.on``) instrument,
    plus its sensor depth offset, plus a fixed margin
    (``distance.above.bottom.file.cut`` in the R source, default 0.15 m --
    a hardcoded function default there too, not an ``init.cops.dat`` field)
    accounting for the file having been trimmed slightly above the true
    bottom to avoid self-shading.
    """
    depth = np.asarray(depth_is_on_depth, dtype=float)
    kept = np.asarray(kept, dtype=bool)
    return float(np.max(depth[kept])) + delta_capteur_m + distance_above_bottom_m


@dataclass(frozen=True)
class BottomReflectanceResult:
    """Bottom depth and substrate reflectance for one instrument (LuZ or EuZ)."""

    bottom_depth: float
    depth_over_bottom: float  # how far the "near-bottom" reading actually sits from bottom_depth
    rb: np.ndarray  # reflectance ~0.30 m above the bottom (per wavelength)
    rb_extrapolated: np.ndarray  # reflectance LOESS-extrapolated all the way to bottom_depth


def compute_bottom_reflectance(
    instrument: str,
    waves: np.ndarray,
    instrument_fit: InstrumentFit,
    edz_fit: InstrumentFit,
    bottom_depth: float,
    span: float,
) -> BottomReflectanceResult:
    """Substrate reflectance for ``instrument`` ("LuZ" or "EuZ"), port of ``compute.bottom.R``.

    ``span`` is ``init.cops.dat``'s ``depth.interval.for.smoothing.optics``
    for ``instrument`` -- the same span already used to fit ``instrument_fit``
    itself, reused here for the secondary (ratio-profile) smoothing pass.
    """
    if instrument not in _BOTTOM_INSTRUMENTS:
        raise ValueError(f"instrument must be one of {_BOTTOM_INSTRUMENTS}, got {instrument!r}")

    waves = np.asarray(waves, dtype=float)
    depth_grid = instrument_fit.depth_grid
    n_waves = len(waves)

    edz_on_grid = np.column_stack(
        [np.interp(depth_grid, edz_fit.depth_grid, edz_fit.aop_fitted[:, i]) for i in range(n_waves)]
    )
    r_z = instrument_fit.aop_fitted * _PI_FACTOR[instrument] / edz_on_grid

    ix_near_bottom = int(np.argmin(np.abs(depth_grid - (bottom_depth - _NEAR_BOTTOM_MARGIN_M))))
    depth_over_bottom = bottom_depth - depth_grid[ix_near_bottom]

    extended_grid = np.append(depth_grid, bottom_depth)
    ix_bottom = len(extended_grid) - 1

    fitted = fit_profile_loess(
        waves,
        depth_grid,
        r_z,
        span,
        extended_grid,
        idx_depth_0=0,
        span_wave_correction=False,
        depth_span=True,
        minimum_obs=10,
    )

    return BottomReflectanceResult(
        bottom_depth=bottom_depth,
        depth_over_bottom=depth_over_bottom,
        rb=fitted.fitted[ix_near_bottom, :],
        rb_extrapolated=fitted.fitted[ix_bottom, :],
    )
