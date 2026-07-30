"""Run the whole per-cast pipeline: fit every depth-profiled instrument
present, self-shading-correct LuZ/EuZ where possible, and derive Rrs/Lw.

Ties :func:`pycops.processing.cast_fit.fit_ed0_for_cast` and
:func:`pycops.processing.cast_fit.fit_cast` together across all instruments
in one cast, applies :func:`pycops.processing.shadow.shadow_correction` to
LuZ/EuZ when the cast carries enough information for it, and computes
:func:`pycops.processing.rrs.compute_rrs` (including nLw), the Forel-Ule/QWIP
diagnostics, and (when EuZ is present) ``Ed0.0m``/``R.0m`` (see
:mod:`pycops.processing.ed0_0m`). Equivalent to one iteration of
``process.cops.R``'s per-cast loop followed by ``compute.aops.R``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import xarray as xr

from pycops.processing.attenuation import compute_K, kd_at_light_fraction
from pycops.processing.bioshade import BioShadeResult
from pycops.processing.bottom import BottomReflectanceResult, compute_bottom_depth, compute_bottom_reflectance
from pycops.processing.cast_fit import InstrumentFit, fit_cast, fit_ed0_for_cast
from pycops.processing.ed0 import Ed0Fit
from pycops.processing.ed0_0m import compute_ed0_subsurface
from pycops.processing.gregg_carder import gregg_carder_diffuse_direct
from pycops.processing.par import par_profile, par_quanta
from pycops.processing.position import PositionOverride
from pycops.processing.qfactor import compute_q_factor
from pycops.processing.qwip import QWIPResult, compute_qwip
from pycops.processing.rrs import RrsResult, compute_rrs
from pycops.processing.shadow import ShadowCorrectionResult, resolve_absorption, shadow_correction
from pycops.processing.solar import sun_position

_DEPTH_PROFILED_INSTRUMENTS = ("EdZ", "LuZ", "EuZ")
_SHADOWABLE_INSTRUMENTS = ("LuZ", "EuZ")

# select.cops.dat's "method" column names which Rrs a researcher already vetted
# as the right one for a given cast (see discovery.CastSelection.method) --
# some casts fit better with the LOESS surface value, others with the linear
# one (e.g. a non-monotonic depth start near the surface fails the linear
# fit's Kolmogorov-Smirnov gate but the LOESS fit tolerates it fine).
_METHOD_LOESS = "Rrs.0p"
_METHOD_LINEAR = "Rrs.0p.linear"

# ds.attrs name (set by discovery.read_one_cast, straight from info.cops.dat's CastInfo) ->
# the init.cops.dat per-instrument key it overrides.
_OVERRIDE_ATTR_TO_INIT_KEY = {
    "sub_surface_removed_layer": "sub.surface.removed.layer.optics",
    "tiltmax": "tiltmax.optics",
    "depth_interval_for_smoothing": "depth.interval.for.smoothing.optics",
    "linear_r2_threshold": "linear.fit.Rsquared.threshold.optics",
    "linear_max_delta_depth": "linear.fit.max.delta.depth.optics",
}


def _apply_info_overrides(init: dict[str, object], ds: xr.Dataset) -> dict[str, object]:
    """Merge ``info.cops.dat``'s per-cast override fields (if any) into a copy of ``init``.

    Port of ``process.cops.R``'s per-cast merge (e.g. ``cops.init$tiltmax.optics <-
    tiltmax.optics``): each override, when present, replaces the *whole* per-instrument dict for
    that key (one value per ``instruments.optics`` entry, same shape ``init`` itself uses), not
    just a single instrument's value -- matching ``CastInfo``'s own one-array-per-field shape.
    Was previously read and editable in the UI but never actually applied here, so a researcher
    adjusting these in ``info.cops.dat`` had no effect at all until this fix.
    """
    overrides = {
        attr: ds.attrs[attr] for attr in _OVERRIDE_ATTR_TO_INIT_KEY if ds.attrs.get(attr) is not None
    }
    if not overrides:
        return init
    instruments = tuple(init["instruments.optics"])
    merged = dict(init)
    for attr, values in overrides.items():
        merged[_OVERRIDE_ATTR_TO_INIT_KEY[attr]] = dict(zip(instruments, values))
    return merged


@dataclass(frozen=True)
class CastResult:
    """Full result of processing one cast: every instrument's fit plus Rrs/Lw."""

    waves: np.ndarray
    ed0_fit: Ed0Fit
    instrument_fits: dict[str, InstrumentFit]
    shadow_corrections: dict[str, ShadowCorrectionResult]  # by instrument ("LuZ"/"EuZ"), whichever succeeded
    shadow_correction_note: str | None  # why shadow correction wasn't applied to some/all instruments, if so
    rrs_loess: RrsResult | None  # from LuZ's (or EuZ-derived) LOESS-fitted surface value
    rrs_linear: RrsResult | None  # from LuZ's (or EuZ-derived) linear-fitted surface value
    rrs_method: str | None  # select.cops.dat's method for this cast, if known
    recommended_rrs: RrsResult | None  # rrs_loess or rrs_linear per rrs_method, whichever is available
    rrs_source: str | None  # "LuZ" or "EuZ" (via the Q factor) -- which instrument Rrs came from, if any
    qwip_loess: QWIPResult | None  # QWIP/Forel-Ule diagnostics on rrs_loess, if available
    qwip_linear: QWIPResult | None  # QWIP/Forel-Ule diagnostics on rrs_linear, if available
    ed0_0m: np.ndarray | None  # Ed0(0-), diffuse/direct-decomposed -- diagnostic only, Rrs doesn't need it
    r0m_loess: np.ndarray | None  # EuZ.0m(LOESS) / Ed0.0m -- subsurface irradiance reflectance
    r0m_linear: np.ndarray | None  # EuZ.0m(linear) / Ed0.0m
    kd_1pct: np.ndarray | None  # mean Kd from surface to the 1% light level, EdZ only
    kd_10pct: np.ndarray | None  # mean Kd from surface to the 10% light level
    kd_pd: np.ndarray | None  # mean Kd from surface to the penetration depth (1/e light level)
    par_0: float | None  # broadband PAR (uEin.m-2.s-1) of Ed0's smoothed surface reference
    par_d_profile: np.ndarray | None  # PAR(z), one value per EdZ depth-grid point
    par_u_profile: np.ndarray | None  # PAR(z) from EuZ (or LuZ*Q.sun.nadir), aligned onto EdZ's grid
    kz_par: np.ndarray | None  # local Kd(PAR), aligned with EdZ's depth_grid[1:]
    k0_par: np.ndarray | None  # depth-integrated Kd(PAR), same alignment as kz_par
    kd_par_1pct: float | None  # mean Kd(PAR) from surface to the 1% light level
    kd_par_10pct: float | None  # mean Kd(PAR) from surface to the 10% light level
    kd_par_pd: float | None  # mean Kd(PAR) from surface to the penetration depth
    bottom_reflectance: dict[str, BottomReflectanceResult]  # by instrument ("LuZ"/"EuZ"), only for SHALLOW casts
    bottom_note: str | None  # why bottom_reflectance is empty despite a SHALLOW-flagged cast, if so
    resolved_longitude: float | None  # position actually used for shadow correction/Ed0.0m, if resolved
    resolved_latitude: float | None  # (may differ from ds.attrs -- e.g. a position_override or GPS file)
    resolved_sun_zenith_deg: float | None  # sun zenith angle (degrees) at the cast's mean time/position
    excluded_wavelengths: tuple[float, ...] = ()  # final-Rrs bands manually NaN'd out (see io.exclusions)


def _mask_rrs_wavelengths(rrs: RrsResult, mask: np.ndarray) -> RrsResult:
    """Return a copy of ``rrs`` with every wavelength where ``mask`` is ``True`` set to ``NaN``
    in ``lw_0p``/``rrs_0p``/``nlw_0p`` -- a final, post-fit QC override (Simon: a band can be bad
    for a specific cast, e.g. UV noise at 380 nm, regardless of which method produced it), not a
    re-fit: whichever method (LOESS/linear) ends up chosen, the excluded band stays ``NaN``."""
    if not np.any(mask):
        return rrs
    lw_0p = np.where(mask, np.nan, rrs.lw_0p)
    rrs_0p = np.where(mask, np.nan, rrs.rrs_0p)
    nlw_0p = np.where(mask, np.nan, rrs.nlw_0p) if rrs.nlw_0p is not None else None
    return replace(rrs, lw_0p=lw_0p, rrs_0p=rrs_0p, nlw_0p=nlw_0p)


def _cast_sun_geometry(
    ds: xr.Dataset, lon: float, lat: float, utc_time_override: pd.Timestamp | None = None
) -> tuple[int, float]:
    """Julian day and sun zenith angle (degrees) for a cast's mean scan time and position.

    ``utc_time_override``, if given (see :class:`~pycops.processing.position.PositionOverride`),
    replaces the cast's own recorded mean scan time -- for a cast whose clock is known to be
    wrong (e.g. no GPS to sync it in the field).
    """
    mean_time = pd.Timestamp(utc_time_override) if utc_time_override is not None else pd.Series(np.asarray(ds["time"].values)).mean()
    hour_utc = mean_time.hour + mean_time.minute / 60.0 + mean_time.second / 3600.0
    zenith_deg, _ = sun_position(mean_time.month, mean_time.day, hour_utc, lon, lat)
    return mean_time.dayofyear, zenith_deg


def _resolve_sun_geometry(
    ds: xr.Dataset, position_override: PositionOverride | None
) -> tuple[float | None, float | None, int | None, float | None, str | None]:
    """Cast position and sun geometry -- independent of ``chl``, shared by shadow correction and Ed0.0m.

    Returns ``(lon, lat, julian_day, sun_zenith_deg, note)``; ``note`` explains why geometry
    couldn't be resolved (``None`` on success). Matches ``derived.data.R``'s ``sunzen``/position
    computation, which likewise doesn't depend on the ``chl`` field at all.
    """
    position_override = position_override or PositionOverride()
    lon = position_override.longitude if position_override.longitude is not None else ds.attrs.get("longitude")
    lat = position_override.latitude if position_override.latitude is not None else ds.attrs.get("latitude")
    if lon is None or lat is None:
        return None, None, None, None, "cast position (longitude/latitude) unavailable"

    julian_day, sun_zenith_deg = _cast_sun_geometry(ds, lon, lat, position_override.utc_time)
    if sun_zenith_deg < 0:
        return lon, lat, julian_day, sun_zenith_deg, "sun below the horizon for this cast"

    return lon, lat, julian_day, sun_zenith_deg, None


def _shadow_correct_instruments(
    ds: xr.Dataset,
    init: dict[str, object],
    waves: np.ndarray,
    instrument_fits: dict[str, InstrumentFit],
    ed0_0p: np.ndarray,
    absorption_waves: np.ndarray | None,
    absorption_values: np.ndarray | None,
    bioshade: BioShadeResult | None,
    lon: float | None,
    lat: float | None,
    julian_day: int | None,
    sun_zenith_deg: float | None,
    geometry_note: str | None,
) -> tuple[dict[str, ShadowCorrectionResult], str | None]:
    if geometry_note is not None:
        return {}, f"{geometry_note}: shadow correction not applied"

    chl = ds.attrs.get("chl_flag")
    if chl is None or (isinstance(chl, float) and np.isnan(chl)):
        return {}, "chl unavailable or NA (info.cops.dat): shadow correction not applied"

    results: dict[str, ShadowCorrectionResult] = {}
    note: str | None = None
    for instrument in _SHADOWABLE_INSTRUMENTS:
        if instrument not in instrument_fits or "EdZ" not in instrument_fits:
            continue
        try:
            absorption = resolve_absorption(
                instrument,
                chl,
                instrument_fits,
                waves,
                absorption_waves=absorption_waves,
                absorption_values=absorption_values,
            )
        except (NotImplementedError, ValueError) as exc:
            note = str(exc)
            continue
        if absorption is None:
            continue

        radius_m = init["radius.instrument.optics"][instrument]
        results[instrument] = shadow_correction(
            instrument=instrument,
            waves=waves,
            absorption=absorption,
            radius_m=radius_m,
            sun_zenith_deg=sun_zenith_deg,
            julian_day=julian_day,
            lon=lon,
            lat=lat,
            ed0_0p=ed0_0p,
            bioshade=bioshade,
        )

    return results, note


def process_cast(
    ds: xr.Dataset,
    init: dict[str, object],
    absorption_waves: np.ndarray | None = None,
    absorption_values: np.ndarray | None = None,
    bioshade: BioShadeResult | None = None,
    position_override: PositionOverride | None = None,
    excluded_wavelengths: Sequence[float] | None = None,
) -> CastResult:
    """Fit Ed0 plus every depth-profiled instrument present in ``ds``, shadow-correct, and Rrs/Lw.

    ``ds`` is one cast from :func:`pycops.io.raw.read_cast` (or
    :func:`pycops.io.discovery.read_deployment_casts`); ``init`` is the parsed
    ``init.cops.dat`` dict from :func:`pycops.io.config.read_init_cops`.
    Instruments the R package's ``instruments.optics`` lists but that aren't
    actually in ``ds`` (e.g. no EuZ sensor on that deployment) are skipped.

    ``ds.attrs["time_window"]`` (an ``info.cops.dat`` per-cast override, set by
    :func:`~pycops.io.discovery.read_deployment_casts`), if present, restricts
    every instrument's kept scans to that elapsed-seconds-from-first-scan
    window, falling back to ``init["time.window"]`` (the deployment-wide
    default) when the per-cast override is absent -- matching
    ``derived.data.R``'s ``Depth.good <- Depth.good & dates.good``, applied
    once per cast before any per-instrument fitting (see
    :func:`pycops.processing.depth.time_window_mask`).

    Shadow correction (see :mod:`pycops.processing.shadow`) requires ``ds`` to
    carry ``chl_flag``/``longitude``/``latitude`` attrs (as
    :func:`~pycops.io.discovery.read_deployment_casts` sets from
    ``info.cops.dat``) and, when ``chl_flag == 0``, ``absorption_waves``/
    ``absorption_values`` from ``absorption.cops.dat`` (see
    :func:`pycops.io.config.read_absorption_cops`/``absorption_for_cast``).
    Whenever any of that is missing, or ``chl_flag`` is a positive chlorophyll
    value (not yet ported), shadow correction is skipped for the affected
    instrument(s) and ``shadow_correction_note`` explains why -- ``rrs_loess``/
    ``rrs_linear`` then fall back to the uncorrected LuZ (or EuZ-derived)
    surface values.

    ``bioshade``, if given, is a :func:`~pycops.processing.bioshade.process_bioshade`
    result from a BioShade shadow-band cast (``select.cops.dat`` flag ``2``) in
    the same deployment folder as ``ds``; it's the caller's job to find and
    process that sibling cast (e.g. via :func:`~pycops.io.discovery.discover_deployment`)
    since ``process_cast`` only ever sees one cast at a time. When present, it
    replaces the Gregg & Carder clear-sky estimate with the measured
    diffuse/direct split for every shadow-corrected instrument on this cast.

    ``position_override``, if given, supplies longitude/latitude/UTC-time
    directly instead of relying on ``ds.attrs``/``ds["time"]`` -- for a cast
    whose own GPS (and any GPS file) is unavailable or known to be wrong, a
    real recurring field situation. Any field left ``None`` on the override
    falls back to the normal source for that field alone. Whatever
    longitude/latitude actually got used (from the override, ``ds.attrs``, or
    -- when the caller is :func:`~pycops.processing.deployment.process_deployment`
    -- a GPS file) is recorded on ``CastResult.resolved_longitude``/
    ``.resolved_latitude`` (``None`` if position couldn't be resolved at all),
    since it can differ from ``ds.attrs`` and callers persisting this result
    (see :mod:`pycops.io.netcdf`) need the position actually used, not just
    what ``info.cops.dat`` originally said. Position/sun-geometry resolution
    doesn't depend on ``chl_flag`` (matching ``derived.data.R``), so
    ``resolved_longitude``/``.resolved_latitude``/``.resolved_sun_zenith_deg``
    can be set even when ``shadow_correction_note`` explains shadow correction
    itself was skipped for an unrelated (``chl``-only) reason.

    Rrs comes from LuZ when present (``rrs_source == "LuZ"``); for a cast with
    EuZ but no LuZ sensor, it's derived instead as ``LuZ.0m = EuZ.0m /
    Q.sun.nadir`` (``rrs_source == "EuZ"``, see
    :func:`pycops.processing.qfactor.compute_q_factor` -- ``Q.sun.nadir`` is
    the constant ``pi`` for every ``chl_flag`` pycops's shadow correction
    already supports; a genuine ``chl_flag > 0`` isn't ported, in which case
    Rrs is left ``None`` rather than raising, same as a cast with neither
    instrument). Shadow correction, when available, is applied to whichever
    instrument Rrs is derived from before the Q-factor conversion.

    If ``ds`` came from :func:`~pycops.io.discovery.read_deployment_casts`, its
    ``rrs_method`` attr (from ``select.cops.dat``) picks ``recommended_rrs``
    between ``rrs_loess`` and ``rrs_linear``, falling back to whichever is
    available if the preferred one is missing entirely (e.g. no LuZ or EuZ)
    *or* present but unusable -- every wavelength ``NaN`` (e.g. the linear
    surface fit's Kolmogorov-Smirnov/R² gate rejected every candidate window
    for this cast, even though ``select.cops.dat`` recorded the linear
    method as the researcher's general preference for the station).

    ``excluded_wavelengths``, if given, is a final, post-fit QC override (see
    :mod:`pycops.io.exclusions` -- a pycops-only feature, no R equivalent):
    every wavelength in it gets NaN'd out of ``rrs_loess``/``rrs_linear`` (and
    their ``lw_0p``/``nlw_0p``), regardless of which one ``recommended_rrs``
    ends up picking, matching within ``waves`` to 1e-6 nm. Recorded verbatim
    on ``CastResult.excluded_wavelengths`` for display/export.

    Each available ``RrsResult`` also gets its normalized water-leaving
    radiance (``nlw_0p``, from ``init.cops.dat``'s ``bandwidth``) and, on
    ``CastResult``, its Forel-Ule/QWIP quality-control diagnostics
    (``qwip_loess``/``qwip_linear``, see
    :func:`pycops.processing.qwip.compute_qwip`) -- ``None`` when the
    corresponding Rrs is unavailable.

    When EuZ is present and position/sun-geometry resolve, ``CastResult``
    also gets ``ed0_0m`` (``Ed0(0-)``, diffuse/direct-decomposed -- see
    :mod:`pycops.processing.ed0_0m`) and ``r0m_loess``/``r0m_linear``
    (subsurface irradiance reflectance, ``EuZ.0m / Ed0.0m``); diagnostic
    only, since Rrs itself only ever needs ``Ed0.0p``. The diffuse/direct
    split reuses whichever of LuZ's/EuZ's shadow correction succeeded (LuZ
    preferred, matching ``compute.aops.R``'s block order when both are
    present), or a fresh Gregg & Carder clear-sky estimate when neither did
    -- unlike shadow correction, this doesn't require ``chl_flag``.

    When EdZ is present, ``CastResult`` also gets ``kd_1pct``/``kd_10pct``/``kd_pd`` -- mean
    diffuse attenuation from the surface down to the 1%/10%/penetration-depth (1/e) light levels
    (see :func:`pycops.processing.attenuation.kd_at_light_fraction`), a port of
    ``generate.cops.DB.R``'s inline Kd computation used for the mission-wide database export.
    Uses ``ed0_0m`` (subsurface Ed0) as the reference when available, matching R exactly;
    otherwise falls back to ``Ed0.0p`` (``ed0_fit.value_at_0``) rather than leaving these ``None``
    for every EuZ-less cast -- a documented deviation, since R always requires ``Ed0.0m``.

    The same ``EdZ``-present gate also computes broadband PAR (port of
    ``compute.PAR.fitted.R``, see :mod:`pycops.processing.par`): ``par_0`` (a single scalar, from
    Ed0's smoothed surface-reference spectrum -- pycops fits Ed0 at one point, never a depth
    profile, unlike R, so there's no ``PAR.0(z)`` curve here; the per-scan illumination-change
    diagnostic that R plot doubles as is already covered by the analyze tab's separate "Ed0
    stability" section), ``par_d_profile`` (PAR at every ``EdZ`` depth-grid point), ``par_u_profile``
    (from EuZ, or ``LuZ * Q.sun.nadir`` when only LuZ is present, interpolated onto EdZ's own depth
    grid the same way :func:`pycops.processing.bottom.compute_bottom_reflectance` already aligns
    one instrument's fit onto another's), and ``kz_par``/``k0_par`` (local/depth-integrated Kd(PAR),
    via :func:`pycops.processing.attenuation.compute_K` treating PAR as a single band) plus
    ``kd_par_1pct``/``kd_par_10pct``/``kd_par_pd`` (the same three light-level fractions as the
    spectral Kd above, via ``kd_at_light_fraction``).

    When ``ds`` is flagged ``SHALLOW`` (``select.cops.dat``'s 4th field,
    ``"1"`` -- see :attr:`~pycops.io.discovery.CastSelection.shallow`),
    ``CastResult.bottom_reflectance`` also gets one
    :class:`~pycops.processing.bottom.BottomReflectanceResult` per available
    instrument (``"LuZ"``/``"EuZ"``), estimating substrate reflectance near
    and extrapolated to the cast's own maximum recorded depth (see
    :mod:`pycops.processing.bottom`) -- ``bottom_note`` explains why it's
    empty if flagged ``SHALLOW`` but EdZ or the ``depth.is.on`` reference
    instrument isn't available.
    """
    init = _apply_info_overrides(init, ds)
    waves = ds["wavelength"].values
    time_window = ds.attrs.get("time_window") or (
        tuple(init["time.window"]) if "time.window" in init else None
    )
    ed0_fit = fit_ed0_for_cast(ds, init, time_window=time_window)

    instrument_fits = {
        instr: fit_cast(ds, init, instr, ed0_fit, time_window=time_window)
        for instr in _DEPTH_PROFILED_INSTRUMENTS
        if instr in ds
    }

    lon, lat, julian_day, sun_zenith_deg, geometry_note = _resolve_sun_geometry(ds, position_override)

    shadow_corrections, shadow_correction_note = _shadow_correct_instruments(
        ds,
        init,
        waves,
        instrument_fits,
        ed0_fit.value_at_0,
        absorption_waves,
        absorption_values,
        bioshade,
        lon,
        lat,
        julian_day,
        sun_zenith_deg,
        geometry_note,
    )

    rrs_loess = rrs_linear = rrs_source = None
    if "LuZ" in instrument_fits:
        rrs_source = "LuZ"
        luz_fit = instrument_fits["LuZ"]
        luz_value_at_0 = luz_fit.value_at_0
        luz_value_at_surface = luz_fit.surface_linear.value_at_surface
        luz_shadow = shadow_corrections.get("LuZ")
        if luz_shadow is not None:
            luz_value_at_0 = luz_value_at_0 / luz_shadow.correction
            luz_value_at_surface = luz_value_at_surface / luz_shadow.correction
    elif "EuZ" in instrument_fits:
        try:
            q_factor = compute_q_factor(ds.attrs.get("chl_flag"), len(waves))
        except NotImplementedError:
            luz_value_at_0 = luz_value_at_surface = None
        else:
            rrs_source = "EuZ"
            euz_fit = instrument_fits["EuZ"]
            euz_value_at_0 = euz_fit.value_at_0
            euz_value_at_surface = euz_fit.surface_linear.value_at_surface
            euz_shadow = shadow_corrections.get("EuZ")
            if euz_shadow is not None:
                euz_value_at_0 = euz_value_at_0 / euz_shadow.correction
                euz_value_at_surface = euz_value_at_surface / euz_shadow.correction
            luz_value_at_0 = euz_value_at_0 / q_factor
            luz_value_at_surface = euz_value_at_surface / q_factor
    else:
        luz_value_at_0 = luz_value_at_surface = None

    qwip_loess = qwip_linear = None
    if luz_value_at_0 is not None:
        indice_water = init["indice.water"]
        rau_fresnel = init["rau.Fresnel"]
        bandwidth = init.get("bandwidth")
        rrs_loess = compute_rrs(luz_value_at_0, ed0_fit.value_at_0, indice_water, rau_fresnel, waves, bandwidth)
        rrs_linear = compute_rrs(
            luz_value_at_surface, ed0_fit.value_at_0, indice_water, rau_fresnel, waves, bandwidth
        )
        if excluded_wavelengths:
            exclude_mask = np.any(
                np.isclose(waves[:, None], np.asarray(excluded_wavelengths, dtype=float)[None, :], atol=1e-6),
                axis=1,
            )
            rrs_loess = _mask_rrs_wavelengths(rrs_loess, exclude_mask)
            rrs_linear = _mask_rrs_wavelengths(rrs_linear, exclude_mask)
        if np.any(np.isfinite(rrs_loess.rrs_0p)):
            qwip_loess = compute_qwip(waves, rrs_loess.rrs_0p)
        if np.any(np.isfinite(rrs_linear.rrs_0p)):
            qwip_linear = compute_qwip(waves, rrs_linear.rrs_0p)

    rrs_method = ds.attrs.get("rrs_method")
    preferred, fallback = (rrs_loess, rrs_linear) if rrs_method == _METHOD_LOESS else (rrs_linear, rrs_loess)
    preferred_usable = preferred is not None and np.any(np.isfinite(preferred.rrs_0p))
    recommended_rrs = preferred if preferred_usable else fallback

    ed0_0m = r0m_loess = r0m_linear = None
    if geometry_note is None and "EuZ" in instrument_fits:
        euz_fit = instrument_fits["EuZ"]
        euz_value_at_0 = euz_fit.value_at_0
        euz_value_at_surface = euz_fit.surface_linear.value_at_surface
        euz_shadow = shadow_corrections.get("EuZ")
        if euz_shadow is not None:
            euz_value_at_0 = euz_value_at_0 / euz_shadow.correction
            euz_value_at_surface = euz_value_at_surface / euz_shadow.correction

        fed_dir_source = shadow_corrections.get("LuZ") or shadow_corrections.get("EuZ")
        if fed_dir_source is not None:
            edir, edif = fed_dir_source.edir, fed_dir_source.edif
        else:
            edif, edir = gregg_carder_diffuse_direct(julian_day, lon, lat, waves, sun_zenith_deg, ed0_fit.value_at_0)

        windspeed_ms = init.get("windspeed_ms", 4.0)
        ed0_sub = compute_ed0_subsurface(
            ed0_fit.value_at_0, edir, edif, sun_zenith_deg, windspeed_ms, euz_value_at_0, euz_value_at_surface
        )
        ed0_0m = ed0_sub.ed0_0m
        r0m_loess = ed0_sub.r0m_loess
        r0m_linear = ed0_sub.r0m_linear

    kd_1pct = kd_10pct = kd_pd = None
    if "EdZ" in instrument_fits:
        edz_fit = instrument_fits["EdZ"]
        # R's generate.cops.DB.R always uses Ed0.0m (subsurface, diffuse/direct-decomposed); that
        # needs EuZ + resolved sun geometry, which not every cast has, so this falls back to
        # Ed0.0p (ed0_fit.value_at_0) rather than leaving these NaN for every EuZ-less cast --
        # a deliberate, documented deviation from a literal port.
        ed0_subsurface = ed0_0m if ed0_0m is not None else ed0_fit.value_at_0
        kd_1pct = kd_at_light_fraction(edz_fit.aop_fitted, edz_fit.depth_grid, ed0_subsurface, 0.01)
        kd_10pct = kd_at_light_fraction(edz_fit.aop_fitted, edz_fit.depth_grid, ed0_subsurface, 0.1)
        kd_pd = kd_at_light_fraction(edz_fit.aop_fitted, edz_fit.depth_grid, ed0_subsurface, 1 / np.e)

    par_0 = par_d_profile = par_u_profile = kz_par = k0_par = None
    kd_par_1pct = kd_par_10pct = kd_par_pd = None
    if "EdZ" in instrument_fits:
        edz_fit = instrument_fits["EdZ"]
        par_0 = par_quanta(waves, ed0_fit.value_at_0)
        par_d_profile = par_profile(waves, edz_fit.aop_fitted)

        par_u_fitted = None
        if "EuZ" in instrument_fits:
            par_u_fitted = instrument_fits["EuZ"].aop_fitted
            par_u_depth_grid = instrument_fits["EuZ"].depth_grid
        elif "LuZ" in instrument_fits:
            luz_fit = instrument_fits["LuZ"]
            par_u_fitted = luz_fit.aop_fitted * compute_q_factor(ds.attrs.get("chl_flag"), len(waves))[None, :]
            par_u_depth_grid = luz_fit.depth_grid
        if par_u_fitted is not None:
            # LuZ/EuZ have their own depth grid, separate from EdZ's -- align onto EdZ's grid the
            # same way compute_bottom_reflectance() already does (bottom.py), rather than assuming
            # a shared grid like R's own single-depth-axis convention does.
            par_u_on_edz_grid = np.column_stack(
                [np.interp(edz_fit.depth_grid, par_u_depth_grid, par_u_fitted[:, i]) for i in range(len(waves))]
            )
            par_u_profile = par_profile(waves, par_u_on_edz_grid)

        kz_par_2d, k0_par_2d = compute_K(
            edz_fit.depth_grid, edz_fit.idx_depth_0, np.array([par_0]), par_d_profile[:, None]
        )
        kz_par, k0_par = kz_par_2d[:, 0], k0_par_2d[:, 0]
        kd_par_1pct = kd_at_light_fraction(par_d_profile[:, None], edz_fit.depth_grid, np.array([par_0]), 0.01)[0]
        kd_par_10pct = kd_at_light_fraction(par_d_profile[:, None], edz_fit.depth_grid, np.array([par_0]), 0.1)[0]
        kd_par_pd = kd_at_light_fraction(
            par_d_profile[:, None], edz_fit.depth_grid, np.array([par_0]), 1 / np.e
        )[0]

    bottom_reflectance: dict[str, BottomReflectanceResult] = {}
    bottom_note: str | None = None
    if ds.attrs.get("shallow"):
        depth_is_on = init.get("depth.is.on")
        if "EdZ" not in instrument_fits:
            bottom_note = "EdZ unavailable: bottom reflectance not computed"
        elif depth_is_on not in instrument_fits:
            bottom_note = f"depth reference instrument ({depth_is_on!r}) unavailable: bottom reflectance not computed"
        else:
            depth_ref = ds[f"{depth_is_on}_Depth"].values
            bottom_depth = compute_bottom_depth(
                depth_ref, instrument_fits[depth_is_on].kept, init["delta.capteur.optics"][depth_is_on]
            )
            edz_fit = instrument_fits["EdZ"]
            for instrument in ("LuZ", "EuZ"):
                if instrument not in instrument_fits:
                    continue
                span = init["depth.interval.for.smoothing.optics"][instrument]
                bottom_reflectance[instrument] = compute_bottom_reflectance(
                    instrument, waves, instrument_fits[instrument], edz_fit, bottom_depth, span
                )
            if not bottom_reflectance:
                bottom_note = "neither LuZ nor EuZ available: bottom reflectance not computed"

    return CastResult(
        waves=waves,
        ed0_fit=ed0_fit,
        instrument_fits=instrument_fits,
        shadow_corrections=shadow_corrections,
        shadow_correction_note=shadow_correction_note,
        rrs_loess=rrs_loess,
        rrs_linear=rrs_linear,
        rrs_method=rrs_method,
        recommended_rrs=recommended_rrs,
        rrs_source=rrs_source,
        qwip_loess=qwip_loess,
        qwip_linear=qwip_linear,
        ed0_0m=ed0_0m,
        r0m_loess=r0m_loess,
        r0m_linear=r0m_linear,
        kd_1pct=kd_1pct,
        kd_10pct=kd_10pct,
        kd_pd=kd_pd,
        par_0=par_0,
        par_d_profile=par_d_profile,
        par_u_profile=par_u_profile,
        kz_par=kz_par,
        k0_par=k0_par,
        kd_par_1pct=kd_par_1pct,
        kd_par_10pct=kd_par_10pct,
        kd_par_pd=kd_par_pd,
        bottom_reflectance=bottom_reflectance,
        bottom_note=bottom_note,
        resolved_longitude=lon,
        resolved_latitude=lat,
        resolved_sun_zenith_deg=sun_zenith_deg,
        excluded_wavelengths=tuple(excluded_wavelengths) if excluded_wavelengths else (),
    )
