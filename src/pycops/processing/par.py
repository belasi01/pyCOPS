"""Photosynthetically Available Radiation (PAR), from an already-fitted spectral profile.

A *scoped* port of the wavelength-integration half of ``compute.PAR.fitted.R`` (photon-flux
weighted, 400-700 nm, Planck's-constant conversion from irradiance to quanta) -- not the full
whole-profile ``PAR.0``/``PAR.d``/``PAR.u`` computation R does at every depth grid point, since
the only thing needed so far is "what fraction of the surface's PAR reaches a given depth" (the
analyze tab's benthic-PAR diagnostic for ``SHALLOW`` casts, see ``pycops.ui.analyze_app``).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

# Planck's constant (J.s), speed of light (m/s), Avogadro's number (mol^-1) -- same SI constants
# compute.PAR.fitted.R uses to convert irradiance (W) to photon flux (quanta).
_H = 6.62607004e-34
_C = 299792458.0
_AV = 6.022140857e23
_PAR_WAVES_NM = np.arange(400, 701, dtype=float)


def _par_quanta(waves_nm: np.ndarray, values: np.ndarray) -> float:
    """Photon-flux-weighted PAR for one spectrum (arbitrary units -- cancel out in a ratio)."""
    order = np.argsort(waves_nm)
    values = np.nan_to_num(values[order], nan=0.0)  # matches compute.PAR.fitted.R's EdZ[is.na(EdZ)] <- 0
    spline = CubicSpline(waves_nm[order], values, bc_type="natural")
    interpolated = np.clip(spline(_PAR_WAVES_NM), 0, None)  # negative spline artifacts -> 0, matching R's NA->0
    quanta = interpolated * _PAR_WAVES_NM * 1e-9 / (_H * _C)
    return float(np.sum(quanta)) * 1e6 / _AV  # uEinstein.m-2.s-1, matching compute.PAR.fitted.R


def percent_par_at_depth(
    waves_nm: np.ndarray,
    fitted_profile: np.ndarray,
    depth_grid: np.ndarray,
    target_depth: float,
) -> float | None:
    """% of the shallowest fitted depth's PAR remaining at ``target_depth``.

    ``fitted_profile`` is ``(n_depth, n_waves)`` (e.g. a cast's ``EdZ_fitted``); ``depth_grid``
    its depth coordinate. Returns ``None`` if the surface PAR is zero or negative (nothing
    meaningful to take a ratio against).
    """
    waves_nm = np.asarray(waves_nm, dtype=float)
    depth_grid = np.asarray(depth_grid, dtype=float)
    fitted_profile = np.asarray(fitted_profile, dtype=float)

    surface_values = fitted_profile[0, :]
    target_values = np.array(
        [np.interp(target_depth, depth_grid, fitted_profile[:, i]) for i in range(fitted_profile.shape[1])]
    )

    par_surface = _par_quanta(waves_nm, surface_values)
    if par_surface <= 0:
        return None
    par_target = _par_quanta(waves_nm, target_values)
    return 100.0 * par_target / par_surface
