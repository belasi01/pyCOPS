"""Run the whole per-cast pipeline: fit every depth-profiled instrument
present, self-shading-correct LuZ/EuZ where possible, and derive Rrs/Lw.

Ties :func:`pycops.processing.cast_fit.fit_ed0_for_cast` and
:func:`pycops.processing.cast_fit.fit_cast` together across all instruments
in one cast, applies :func:`pycops.processing.shadow.shadow_correction` to
LuZ/EuZ when the cast carries enough information for it, and computes
:func:`pycops.processing.rrs.compute_rrs`. Equivalent to one iteration of
``process.cops.R``'s per-cast loop followed by ``compute.aops.R`` -- except
the Gregg & Carder-based ``Ed0.0m`` diffuse/direct split and Q/f BRDF factors,
so a EuZ-only cast (no LuZ) still can't produce Rrs (that needs the Q/f-based
EuZ-to-LuZ conversion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from pycops.processing.cast_fit import InstrumentFit, fit_cast, fit_ed0_for_cast
from pycops.processing.ed0 import Ed0Fit
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


@dataclass(frozen=True)
class CastResult:
    """Full result of processing one cast: every instrument's fit plus Rrs/Lw."""

    waves: np.ndarray
    ed0_fit: Ed0Fit
    instrument_fits: dict[str, InstrumentFit]
    shadow_corrections: dict[str, ShadowCorrectionResult]  # by instrument ("LuZ"/"EuZ"), whichever succeeded
    shadow_correction_note: str | None  # why shadow correction wasn't applied to some/all instruments, if so
    rrs_loess: RrsResult | None  # from LuZ's LOESS-fitted surface value (shadow-corrected if available)
    rrs_linear: RrsResult | None  # from LuZ's linear-fitted surface value (shadow-corrected if available)
    rrs_method: str | None  # select.cops.dat's method for this cast, if known
    recommended_rrs: RrsResult | None  # rrs_loess or rrs_linear per rrs_method, whichever is available


def _cast_sun_geometry(ds: xr.Dataset, lon: float, lat: float) -> tuple[int, float]:
    """Julian day and sun zenith angle (degrees) for a cast's mean scan time and position."""
    mean_time = pd.Series(np.asarray(ds["time"].values)).mean()
    hour_utc = mean_time.hour + mean_time.minute / 60.0 + mean_time.second / 3600.0
    zenith_deg, _ = sun_position(mean_time.month, mean_time.day, hour_utc, lon, lat)
    return mean_time.dayofyear, zenith_deg


def _shadow_correct_instruments(
    ds: xr.Dataset,
    init: dict[str, object],
    waves: np.ndarray,
    instrument_fits: dict[str, InstrumentFit],
    ed0_0p: np.ndarray,
    absorption_waves: np.ndarray | None,
    absorption_values: np.ndarray | None,
) -> tuple[dict[str, ShadowCorrectionResult], str | None]:
    chl = ds.attrs.get("chl_flag")
    if chl is None or (isinstance(chl, float) and np.isnan(chl)):
        return {}, "chl unavailable or NA (info.cops.dat): shadow correction not applied"

    lon = ds.attrs.get("longitude")
    lat = ds.attrs.get("latitude")
    if lon is None or lat is None:
        return {}, "cast position (longitude/latitude) unavailable: shadow correction not applied"

    julian_day, sun_zenith_deg = _cast_sun_geometry(ds, lon, lat)
    if sun_zenith_deg < 0:
        return {}, "sun below the horizon for this cast: shadow correction not applied"

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
        )

    return results, note


def process_cast(
    ds: xr.Dataset,
    init: dict[str, object],
    absorption_waves: np.ndarray | None = None,
    absorption_values: np.ndarray | None = None,
) -> CastResult:
    """Fit Ed0 plus every depth-profiled instrument present in ``ds``, shadow-correct, and Rrs/Lw.

    ``ds`` is one cast from :func:`pycops.io.raw.read_cast` (or
    :func:`pycops.io.discovery.read_deployment_casts`); ``init`` is the parsed
    ``init.cops.dat`` dict from :func:`pycops.io.config.read_init_cops`.
    Instruments the R package's ``instruments.optics`` lists but that aren't
    actually in ``ds`` (e.g. no EuZ sensor on that deployment) are skipped.

    Shadow correction (see :mod:`pycops.processing.shadow`) requires ``ds`` to
    carry ``chl_flag``/``longitude``/``latitude`` attrs (as
    :func:`~pycops.io.discovery.read_deployment_casts` sets from
    ``info.cops.dat``) and, when ``chl_flag == 0``, ``absorption_waves``/
    ``absorption_values`` from ``absorption.cops.dat`` (see
    :func:`pycops.io.config.read_absorption_cops`/``absorption_for_cast``).
    Whenever any of that is missing, or ``chl_flag`` is a positive chlorophyll
    value (not yet ported), shadow correction is skipped for the affected
    instrument(s) and ``shadow_correction_note`` explains why -- ``rrs_loess``/
    ``rrs_linear`` then fall back to the uncorrected LuZ surface values.

    If ``ds`` came from :func:`~pycops.io.discovery.read_deployment_casts`, its
    ``rrs_method`` attr (from ``select.cops.dat``) picks ``recommended_rrs``
    between ``rrs_loess`` and ``rrs_linear``, falling back to whichever is
    available if the preferred one is missing (e.g. no LuZ) or ``None`` (the
    surface fit failed for every wavelength).
    """
    waves = ds["wavelength"].values
    ed0_fit = fit_ed0_for_cast(ds, init)

    instrument_fits = {instr: fit_cast(ds, init, instr, ed0_fit) for instr in _DEPTH_PROFILED_INSTRUMENTS if instr in ds}

    shadow_corrections, shadow_correction_note = _shadow_correct_instruments(
        ds, init, waves, instrument_fits, ed0_fit.value_at_0, absorption_waves, absorption_values
    )

    rrs_loess = rrs_linear = None
    if "LuZ" in instrument_fits:
        luz_fit = instrument_fits["LuZ"]
        indice_water = init["indice.water"]
        rau_fresnel = init["rau.Fresnel"]

        luz_value_at_0 = luz_fit.value_at_0
        luz_value_at_surface = luz_fit.surface_linear.value_at_surface
        luz_shadow = shadow_corrections.get("LuZ")
        if luz_shadow is not None:
            luz_value_at_0 = luz_value_at_0 / luz_shadow.correction
            luz_value_at_surface = luz_value_at_surface / luz_shadow.correction

        rrs_loess = compute_rrs(luz_value_at_0, ed0_fit.value_at_0, indice_water, rau_fresnel)
        rrs_linear = compute_rrs(luz_value_at_surface, ed0_fit.value_at_0, indice_water, rau_fresnel)

    rrs_method = ds.attrs.get("rrs_method")
    preferred, fallback = (rrs_loess, rrs_linear) if rrs_method == _METHOD_LOESS else (rrs_linear, rrs_loess)
    recommended_rrs = preferred if preferred is not None else fallback

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
    )
