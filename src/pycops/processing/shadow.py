"""Instrument self-shading (shadow) correction.

Ports ``shadow.data.R`` (Gordon & Ding 1992 coefficients), ``shadow.epsilon.R``
(the self-shading error formula), and the absorption-resolution and
orchestration logic of ``shadow.correction.R`` -- except the Morel &
Maritorena chlorophyll-based absorption model (``info.cops.dat``'s ``chl`` a
positive, non-999 value), which is not yet ported. The sky/sun diffuse-direct
split needed here comes from a measured BioShade cast
(:mod:`pycops.processing.bioshade`) when one is available for the deployment,
falling back to the Gregg & Carder clear-sky model
(:mod:`pycops.processing.gregg_carder`) otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycops.processing.bioshade import BioShadeResult
from pycops.processing.cast_fit import InstrumentFit
from pycops.processing.gregg_carder import clear_sky_irradiance

# Gordon & Ding (1992) self-shading coefficients, port of shadow.data.R.
_GORDON_DING_ZENITH = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
_GORDON_DING_SUN = {
    "LuZ": np.array([2.17, 2.17, 2.23, 2.23, 2.29, 2.37, 2.41, 2.45, 2.45, 2.45]),
    "EuZ": np.array([3.14, 3.14, 3.05, 2.94, 2.80, 2.64, 2.47, 2.33, 2.33, 2.33]),
}
_GORDON_DING_SKY = {"LuZ": 4.61, "EuZ": 2.70}

_SHADOWABLE_INSTRUMENTS = ("LuZ", "EuZ")


@dataclass(frozen=True)
class ShadowEpsilon:
    """Self-shading error for the sun and sky (diffuse) light components."""

    eps_sun: np.ndarray
    eps_sky: np.ndarray
    eps: np.ndarray


def shadow_epsilon(
    instrument: str, aR: np.ndarray, sun_zenith_deg: float, ratio_edsky_edsun: np.ndarray
) -> ShadowEpsilon:
    """Self-shading error from absorption*radius (``aR``), sun zenith, and sky/sun irradiance ratio.

    ``instrument`` is ``"LuZ"`` or ``"EuZ"``. Port of ``shadow.epsilon.R``.
    """
    if instrument not in _SHADOWABLE_INSTRUMENTS:
        raise ValueError(f"instrument must be one of {_SHADOWABLE_INSTRUMENTS}, got {instrument!r}")

    aR = np.asarray(aR, dtype=float)
    tan_sunzen_water = np.tan(np.arcsin(np.sin(np.radians(sun_zenith_deg)) / 1.33))
    kprime_sun = np.interp(sun_zenith_deg, _GORDON_DING_ZENITH, _GORDON_DING_SUN[instrument]) / tan_sunzen_water
    kprime_sky = _GORDON_DING_SKY[instrument]

    eps_sun = 1.0 - np.exp(-kprime_sun * aR)
    eps_sky = 1.0 - np.exp(-kprime_sky * aR)
    ratio = np.asarray(ratio_edsky_edsun, dtype=float)
    eps = (eps_sun + eps_sky * ratio) / (1.0 + ratio)

    return ShadowEpsilon(eps_sun=eps_sun, eps_sky=eps_sky, eps=eps)


@dataclass(frozen=True)
class AbsorptionResult:
    """Resolved per-wavelength total absorption, ready for the ``aR`` shadow term."""

    waves: np.ndarray
    values: np.ndarray  # absorption coefficient a(lambda), 1/m
    source: str  # "file", "kd", or "chlorophyll" (not yet implemented)


def kd_derived_absorption(instrument: str, instrument_fits: dict[str, InstrumentFit]) -> np.ndarray:
    """Absorption from Morel & Maritorena (2001) eq. 8', using this cast's own Kd/R fits.

    Prefers the linear-fit Kd/surface values (``K.EdZ.surf``, ``<instrument>.0m.linear``);
    falls back to the LOESS ones (``K0.EdZ.fitted`` near 2.5 m depth, ``<instrument>.0m``)
    at wavelengths where either the linear EdZ or ``instrument`` Kd is missing. Reproduces
    ``shadow.correction.R``'s indexing quirk where the depth-grid index nearest 2.5 m is
    reused directly as a row index into ``K0.EdZ.fitted`` (whose rows are offset by one
    relative to the depth grid) -- kept as-is for numerical fidelity with the R package.

    ``instrument_fits`` is :attr:`pycops.processing.process_cast.CastResult.instrument_fits`
    (must contain ``"EdZ"`` and ``instrument``).
    """
    if instrument not in _SHADOWABLE_INSTRUMENTS:
        raise ValueError(f"instrument must be one of {_SHADOWABLE_INSTRUMENTS}, got {instrument!r}")

    edz_fit = instrument_fits["EdZ"]
    instr_fit = instrument_fits[instrument]

    value_at_0 = instr_fit.value_at_0
    value_at_0_linear = instr_fit.surface_linear.value_at_surface
    valid = np.isfinite(value_at_0)

    k_edz_linear = edz_fit.surface_linear.k_surf
    k_instr_linear = instr_fit.surface_linear.k_surf

    use_loess = np.any(~np.isfinite(k_edz_linear[valid])) or np.any(~np.isfinite(k_instr_linear[valid]))

    q = 4.0  # approximate Q-factor for sub-surface irradiance reflectance
    if use_loess:
        ix_2_5 = int(np.argmin(np.abs(edz_fit.depth_grid - 2.5)))
        ix_2_5 = min(ix_2_5, len(edz_fit.K0) - 1)
        kd = edz_fit.K0[ix_2_5, :]
        x = 0
        while np.any(~np.isfinite(kd[valid])):
            x += 1
            if ix_2_5 - x < 0:
                raise ValueError("no finite EdZ K0 row found while backing off from 2.5 m depth")
            kd = edz_fit.K0[ix_2_5 - x, :]
        ed_0m = edz_fit.value_at_0
        r = (value_at_0 * q) / ed_0m
    else:
        kd = k_edz_linear
        ed_0m = edz_fit.surface_linear.value_at_surface
        r = (value_at_0_linear * q) / ed_0m

    return kd * 0.9 * (1.0 - r) / (1.0 + 2.25 * r)


def resolve_absorption(
    instrument: str,
    chl: float,
    instrument_fits: dict[str, InstrumentFit],
    waves: np.ndarray,
    absorption_waves: np.ndarray | None = None,
    absorption_values: np.ndarray | None = None,
) -> AbsorptionResult | None:
    """Resolve per-wavelength absorption per ``info.cops.dat``'s ``chl`` field.

    - ``chl`` is ``NaN``: shadow correction is disabled for this cast -- returns ``None``.
    - ``chl == 0``: absorption comes from ``absorption.cops.dat``, pass its per-cast
      ``absorption_waves``/``absorption_values`` row (see
      :func:`pycops.io.config.read_absorption_cops`).
    - ``chl == 999``: absorption is derived from this cast's own fitted Kd (see
      :func:`kd_derived_absorption`); ``instrument_fits`` is
      :attr:`~pycops.processing.process_cast.CastResult.instrument_fits` and ``waves`` is
      the cast's wavelength grid.
    - ``chl > 0`` (an actual chlorophyll concentration): not yet ported (Morel &
      Maritorena chlorophyll-based absorption model, ``popt.R``) -- raises
      ``NotImplementedError``.
    """
    if np.isnan(chl):
        return None

    if chl == 0:
        if absorption_waves is None or absorption_values is None:
            raise ValueError("chl == 0 requires absorption_waves/absorption_values from absorption.cops.dat")
        return AbsorptionResult(waves=np.asarray(absorption_waves, dtype=float), values=np.asarray(absorption_values, dtype=float), source="file")

    if chl == 999:
        values = kd_derived_absorption(instrument, instrument_fits)
        return AbsorptionResult(waves=np.asarray(waves, dtype=float), values=values, source="kd")

    raise NotImplementedError(
        "chlorophyll-based absorption (Morel & Maritorena, info.cops.dat chl > 0) is not yet ported"
    )


@dataclass(frozen=True)
class ShadowCorrectionResult:
    """Full self-shading correction for one instrument on one cast."""

    aR: np.ndarray
    edif: np.ndarray
    edir: np.ndarray
    ratio_edsky_edsun: np.ndarray
    eps_sun: np.ndarray
    eps_sky: np.ndarray
    eps: np.ndarray
    correction: np.ndarray  # divide the shadowed quantity by this to correct it (< 1: signal lost to shading)
    absorption: AbsorptionResult


def shadow_correction(
    instrument: str,
    waves: np.ndarray,
    absorption: AbsorptionResult,
    radius_m: float,
    sun_zenith_deg: float,
    julian_day: int,
    lon: float,
    lat: float,
    ed0_0p: np.ndarray,
    bioshade: BioShadeResult | None = None,
) -> ShadowCorrectionResult:
    """Self-shading correction factor for ``instrument`` at ``waves``.

    ``absorption`` comes from :func:`resolve_absorption`; ``radius_m`` is the
    instrument's radius (``radius.instrument.optics`` in ``init.cops.dat``).

    When ``bioshade`` is given (a processed BioShade cast from the same
    deployment -- see :func:`pycops.processing.bioshade.process_bioshade`),
    the sky/sun diffuse-direct split uses its measured values, matching
    ``shadow.correction.R``'s ``SB`` branch (``Edif = Ed0.dif``,
    ``Edir = Ed0.tot - Edif``); ``julian_day``, ``lon``, ``lat``, and
    ``ed0_0p`` are then unused. Otherwise the split comes from the Gregg &
    Carder clear-sky model, with the same visibility-search
    ``shadow.correction.R`` uses to match the model's total irradiance at
    490 nm to the cast's own measured ``Ed0.0p``.
    """
    waves = np.asarray(waves, dtype=float)
    a_interp = np.interp(waves, absorption.waves, absorption.values)
    aR = a_interp * radius_m

    if bioshade is not None:
        edif = np.interp(waves, bioshade.waves, bioshade.ed0_dif)
        ed0_tot = np.interp(waves, bioshade.waves, bioshade.ed0_tot)
        edir = ed0_tot - edif
    else:
        ix_490 = int(np.argmin(np.abs(waves - 490.0)))
        ed0_0p = np.asarray(ed0_0p, dtype=float)

        visibility = 25.0
        egc = clear_sky_irradiance(julian_day, lon, lat, waves, sun_zenith_deg, visibility_km=visibility)
        ratio = egc.ed[ix_490] * 100.0 / ed0_0p[ix_490]
        while ratio > 1.05 and visibility > 0.5:
            visibility -= 0.5
            egc = clear_sky_irradiance(julian_day, lon, lat, waves, sun_zenith_deg, visibility_km=visibility)
            ratio = egc.ed[ix_490] * 100.0 / ed0_0p[ix_490]

        edif = egc.edif * 100.0
        edir = egc.edir * 100.0

    ratio_edsky_edsun = edif / edir

    epss = shadow_epsilon(instrument, aR, sun_zenith_deg, ratio_edsky_edsun)
    correction = 1.0 - epss.eps

    return ShadowCorrectionResult(
        aR=aR,
        edif=edif,
        edir=edir,
        ratio_edsky_edsun=ratio_edsky_edsun,
        eps_sun=epss.eps_sun,
        eps_sky=epss.eps_sky,
        eps=epss.eps,
        correction=correction,
        absorption=absorption,
    )
