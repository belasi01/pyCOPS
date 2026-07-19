"""``Ed0(0-)``: downwelling irradiance just below the surface, diffuse/direct-corrected.

Port of the "Computing Ed0.0m" block in ``compute.aops.R``, plus ``R.0m``
(subsurface irradiance reflectance, ``EuZ.0m / Ed0.0m``). Diagnostic only --
Rrs itself only ever needs ``Ed0.0p`` (see :mod:`pycops.processing.rrs`), not
this -- so this is unrelated to Rrs's own gaps/fallbacks.

``R.0p`` isn't exposed here: the R source computes it with the *identical*
formula as ``R.0m`` (``EuZ.0m / Ed0.0m`` both times, not
``EuZ.0m.linear``/``Ed0.0p`` or anything that would actually differ) -- an
apparent copy-paste artifact, since ``R.0m.linear``/``R.0p.linear`` are
likewise always identical in the R source. Exposing a separately-named
``r0p`` field here that's provably always equal to ``r0m`` would be
misleading rather than faithful, so it's simply omitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycops.processing.gregg_carder import surface_reflectance


@dataclass(frozen=True)
class Ed0SubsurfaceResult:
    """``Ed0(0-)`` and the subsurface irradiance reflectance derived from it."""

    ed0_0m: np.ndarray
    fed_dir: np.ndarray  # fraction of Ed0(0-) that's direct sunlight (vs. diffuse skylight)
    r0m_loess: np.ndarray  # EuZ.0m(LOESS) / Ed0.0m
    r0m_linear: np.ndarray  # EuZ.0m(linear) / Ed0.0m


def compute_ed0_0m(ed0_0p: np.ndarray, edir: np.ndarray, edif: np.ndarray, sun_zenith_deg: float, windspeed_ms: float) -> np.ndarray:
    """``Ed0(0-)``: split ``Ed0(0+)`` into direct/diffuse (via ``edir``/``edif``) and apply
    each component's own Fresnel surface transmission (:func:`~pycops.processing.gregg_carder.surface_reflectance`).
    """
    ed0_0p = np.asarray(ed0_0p, dtype=float)
    fed_dir = edir / (edir + edif)
    rod, ros = surface_reflectance(sun_zenith_deg, windspeed_ms)
    return ed0_0p * fed_dir * (1.0 - rod) + ed0_0p * (1.0 - fed_dir) * (1.0 - ros)


def compute_ed0_subsurface(
    ed0_0p: np.ndarray,
    edir: np.ndarray,
    edif: np.ndarray,
    sun_zenith_deg: float,
    windspeed_ms: float,
    euz_value_at_0: np.ndarray,
    euz_value_at_surface: np.ndarray,
) -> Ed0SubsurfaceResult:
    """``Ed0(0-)`` plus ``R.0m`` (LOESS and linear) from an already surface-extrapolated EuZ."""
    ed0_0m = compute_ed0_0m(ed0_0p, edir, edif, sun_zenith_deg, windspeed_ms)
    fed_dir = np.asarray(edir, dtype=float) / (np.asarray(edir, dtype=float) + np.asarray(edif, dtype=float))
    return Ed0SubsurfaceResult(
        ed0_0m=ed0_0m,
        fed_dir=fed_dir,
        r0m_loess=np.asarray(euz_value_at_0, dtype=float) / ed0_0m,
        r0m_linear=np.asarray(euz_value_at_surface, dtype=float) / ed0_0m,
    )
