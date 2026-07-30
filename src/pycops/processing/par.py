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


def par_quanta(waves_nm: np.ndarray, values: np.ndarray) -> float:
    """Photon-flux-weighted PAR for one spectrum, in uEinstein.m-2.s-1 when ``values`` is in
    COPS's native calibrated units (uW/cm^2/nm) -- matching ``compute.PAR.fitted.R``'s
    Planck's-constant photon-flux conversion, including its ``* 1E-2`` uW/cm^2/nm -> W/m^2/nm
    SI conversion (confirmed against ``shadow.correction.R``'s matching ``* 100`` in the other
    direction) -- this cancels out in a ratio (e.g. :func:`percent_par_at_depth`, Kd(PAR)), which
    is why its earlier omission here went undetected until an absolute PAR value was needed."""
    order = np.argsort(waves_nm)
    values = np.nan_to_num(values[order], nan=0.0)  # matches compute.PAR.fitted.R's EdZ[is.na(EdZ)] <- 0
    spline = CubicSpline(waves_nm[order], values, bc_type="natural")
    interpolated = np.clip(spline(_PAR_WAVES_NM), 0, None)  # negative spline artifacts -> 0, matching R's NA->0
    interpolated = interpolated * 1e-2  # uW/cm^2/nm -> W/m^2/nm (SI), matching compute.PAR.fitted.R
    quanta = interpolated * _PAR_WAVES_NM * 1e-9 / (_H * _C)
    return float(np.sum(quanta)) * 1e6 / _AV  # uEinstein.m-2.s-1, matching compute.PAR.fitted.R


def par_profile(waves_nm: np.ndarray, fitted_profile: np.ndarray) -> np.ndarray:
    """PAR (uEinstein.m-2.s-1) at every depth-grid point of a fitted spectral profile.

    ``fitted_profile`` is ``(n_depth, n_waves)`` (e.g. a cast's ``EdZ_fitted``). Full-profile
    generalization of :func:`percent_par_at_depth`'s two-depth ratio -- port of the per-depth loop
    in ``compute.PAR.fitted.R`` that produces ``PAR.d.fitted``/``PAR.u.fitted``.
    """
    waves_nm = np.asarray(waves_nm, dtype=float)
    fitted_profile = np.asarray(fitted_profile, dtype=float)
    return np.array([par_quanta(waves_nm, fitted_profile[i, :]) for i in range(fitted_profile.shape[0])])


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

    par_surface = par_quanta(waves_nm, surface_values)
    if par_surface <= 0:
        return None
    par_target = par_quanta(waves_nm, target_values)
    return 100.0 * par_target / par_surface
