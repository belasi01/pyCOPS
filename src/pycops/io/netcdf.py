"""Persist a :class:`~pycops.processing.process_cast.CastResult` to NetCDF.

`xarray` + NetCDF (via the ``netCDF4`` engine) was chosen early in the
project as the target in-repo data format -- self-describing, native support
for the ``(time, wavelength, depth)`` structure of a cast, and interoperable
with the R package's own ``ncdf4`` workflow -- but nothing wired it up until
now. :func:`cast_result_to_dataset` packages every array in a ``CastResult``
into one ``xarray.Dataset``; :func:`write_cast_result` writes it to disk.

Each depth-profiled instrument (EdZ/LuZ/EuZ) gets its own depth dimension
(``"<instrument>_depth"``) since :func:`~pycops.processing.depth.depth_grid`
is built per instrument from that instrument's own kept scans and isn't
guaranteed to match another instrument's grid length. ``KZ``/``K0`` are
naturally one row shorter than ``depth_grid`` (they're aligned with
``depth_grid[1:]``, see ``compute_K``); a leading NaN row is prepended so
every per-instrument variable shares the same depth dimension, trading a
little redundancy for a much simpler schema.

``rrs_method``/``rrs_source``/``shadow_correction_note``/``bottom_note``/
``longitude``/``latitude`` (the latter two preferring
``CastResult.resolved_longitude``/``.resolved_latitude`` when position/sun
geometry actually resolved) are always written as global attrs, along with
the QWIP/Forel-Ule scalar diagnostics (``qwip_loess_*``/``qwip_linear_*``)
when available; nLw (``nlw_0p_loess``/``nlw_0p_linear``/``nlw_0p_recommended``),
``ed0_0m``/``r0m_loess``/``r0m_linear`` (see
:mod:`pycops.processing.ed0_0m`), and ``<instrument>_rb``/``_rb_extrapolated``
(see :mod:`pycops.processing.bottom`, plus per-instrument ``bottom_depth``/
``rb_depth_over_bottom`` attrs) are written whenever they were computed (nLw
needs ``init.cops.dat``'s ``bandwidth``, see :mod:`pycops.processing.nlw`;
``ed0_0m``/``r0m_*`` need EuZ present and position/sun-geometry to resolve;
``rb``/``rb_extrapolated`` need the cast flagged ``SHALLOW``).
Passing the original ``ds`` (the cast read by
:func:`pycops.io.raw.read_cast`) is optional but adds real value: the
per-scan boolean ``kept`` mask and Ed0's per-scan illumination ``correction``
get a real ``time`` coordinate instead of a bare integer index, and
``chl_flag``/``qc_flag``/``shallow`` are additionally copied onto the global
attrs, along with ``longitude``/``latitude`` when they weren't otherwise
resolved.

:func:`write_deployment_result` writes every profile cast in a
:class:`~pycops.processing.deployment.DeploymentProcessingResult` (from
:func:`~pycops.processing.deployment.process_deployment`) to its own
``<cast file stem>.nc`` in a target directory -- BioShade casts aren't
written (they have no ``CastResult``, just a ``BioShadeResult``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from pycops.processing.deployment import DeploymentProcessingResult
from pycops.processing.process_cast import CastResult

_DEPTH_PROFILED_INSTRUMENTS = ("EdZ", "LuZ", "EuZ")


def _pad_leading_nan(array: np.ndarray) -> np.ndarray:
    """Prepend one NaN row so a ``(n-1, n_waves)`` array aligns with an ``n``-point depth grid."""
    pad = np.full((1, array.shape[1]), np.nan)
    return np.concatenate([pad, array], axis=0)


def cast_result_to_dataset(cast_result: CastResult, ds: xr.Dataset | None = None) -> xr.Dataset:
    """Build one ``xarray.Dataset`` holding every array in ``cast_result``.

    ``ds``, if given, is the original cast (from :func:`pycops.io.raw.read_cast`
    or :func:`pycops.io.discovery.read_deployment_casts`) -- see the module
    docstring for what it adds.
    """
    waves = np.asarray(cast_result.waves, dtype=float)
    data_vars: dict[str, tuple] = {}
    coords: dict[str, np.ndarray] = {"wavelength": waves}
    attrs: dict[str, object] = {}

    time_dim = "time"
    time_coord = ds["time"].values if ds is not None else np.arange(cast_result.ed0_fit.correction.shape[0])
    coords[time_dim] = time_coord

    data_vars["ed0_value_at_0"] = ("wavelength", cast_result.ed0_fit.value_at_0)
    data_vars["ed0_correction"] = ((time_dim, "wavelength"), cast_result.ed0_fit.correction)

    for instrument, fit in cast_result.instrument_fits.items():
        depth_dim = f"{instrument}_depth"
        coords[depth_dim] = fit.depth_grid
        attrs[f"{instrument}_idx_depth_0"] = fit.idx_depth_0

        data_vars[f"{instrument}_fitted"] = ((depth_dim, "wavelength"), fit.aop_fitted)
        data_vars[f"{instrument}_value_at_0"] = ("wavelength", fit.value_at_0)
        data_vars[f"{instrument}_detection_limit"] = ("wavelength", fit.detection_limit)
        data_vars[f"{instrument}_KZ"] = ((depth_dim, "wavelength"), _pad_leading_nan(fit.KZ))
        data_vars[f"{instrument}_K0"] = ((depth_dim, "wavelength"), _pad_leading_nan(fit.K0))

        linear = fit.surface_linear
        data_vars[f"{instrument}_surface_value_at_surface"] = ("wavelength", linear.value_at_surface)
        data_vars[f"{instrument}_surface_k_surf"] = ("wavelength", linear.k_surf)
        data_vars[f"{instrument}_surface_z_interval"] = ("wavelength", linear.z_interval)
        data_vars[f"{instrument}_surface_ix_z_interval"] = ("wavelength", linear.ix_z_interval)
        data_vars[f"{instrument}_surface_r2"] = ("wavelength", linear.r2)
        data_vars[f"{instrument}_surface_ks_pvalue"] = ("wavelength", linear.ks_pvalue)
        data_vars[f"{instrument}_kept"] = (time_dim, fit.kept.astype(np.int8))

    for instrument, shadow in cast_result.shadow_corrections.items():
        data_vars[f"{instrument}_shadow_aR"] = ("wavelength", shadow.aR)
        data_vars[f"{instrument}_shadow_edif"] = ("wavelength", shadow.edif)
        data_vars[f"{instrument}_shadow_edir"] = ("wavelength", shadow.edir)
        data_vars[f"{instrument}_shadow_ratio_edsky_edsun"] = ("wavelength", shadow.ratio_edsky_edsun)
        data_vars[f"{instrument}_shadow_eps_sun"] = ("wavelength", shadow.eps_sun)
        data_vars[f"{instrument}_shadow_eps_sky"] = ("wavelength", shadow.eps_sky)
        data_vars[f"{instrument}_shadow_eps"] = ("wavelength", shadow.eps)
        data_vars[f"{instrument}_shadow_correction"] = ("wavelength", shadow.correction)
        data_vars[f"{instrument}_absorption"] = ("wavelength", shadow.absorption.values)
        attrs[f"{instrument}_absorption_source"] = shadow.absorption.source

    if cast_result.rrs_loess is not None:
        data_vars["lw_0p_loess"] = ("wavelength", cast_result.rrs_loess.lw_0p)
        data_vars["rrs_0p_loess"] = ("wavelength", cast_result.rrs_loess.rrs_0p)
        if cast_result.rrs_loess.nlw_0p is not None:
            data_vars["nlw_0p_loess"] = ("wavelength", cast_result.rrs_loess.nlw_0p)
    if cast_result.rrs_linear is not None:
        data_vars["lw_0p_linear"] = ("wavelength", cast_result.rrs_linear.lw_0p)
        data_vars["rrs_0p_linear"] = ("wavelength", cast_result.rrs_linear.rrs_0p)
        if cast_result.rrs_linear.nlw_0p is not None:
            data_vars["nlw_0p_linear"] = ("wavelength", cast_result.rrs_linear.nlw_0p)
    if cast_result.recommended_rrs is not None:
        data_vars["lw_0p_recommended"] = ("wavelength", cast_result.recommended_rrs.lw_0p)
        data_vars["rrs_0p_recommended"] = ("wavelength", cast_result.recommended_rrs.rrs_0p)
        if cast_result.recommended_rrs.nlw_0p is not None:
            data_vars["nlw_0p_recommended"] = ("wavelength", cast_result.recommended_rrs.nlw_0p)

    if cast_result.ed0_0m is not None:
        data_vars["ed0_0m"] = ("wavelength", cast_result.ed0_0m)
    if cast_result.r0m_loess is not None:
        data_vars["r0m_loess"] = ("wavelength", cast_result.r0m_loess)
    if cast_result.r0m_linear is not None:
        data_vars["r0m_linear"] = ("wavelength", cast_result.r0m_linear)

    for label, qwip in (("loess", cast_result.qwip_loess), ("linear", cast_result.qwip_linear)):
        if qwip is None:
            continue
        attrs[f"qwip_{label}_avw"] = qwip.avw
        attrs[f"qwip_{label}_ndi"] = qwip.ndi
        attrs[f"qwip_{label}_predicted_ndi"] = qwip.predicted_ndi
        attrs[f"qwip_{label}_score"] = qwip.score
        attrs[f"qwip_{label}_passed"] = int(qwip.passed)
        attrs[f"qwip_{label}_water_class"] = qwip.water_class
        attrs[f"qwip_{label}_fu"] = qwip.fu

    for instrument, bottom in cast_result.bottom_reflectance.items():
        data_vars[f"{instrument}_rb"] = ("wavelength", bottom.rb)
        data_vars[f"{instrument}_rb_extrapolated"] = ("wavelength", bottom.rb_extrapolated)
        attrs[f"{instrument}_bottom_depth"] = bottom.bottom_depth
        attrs[f"{instrument}_rb_depth_over_bottom"] = bottom.depth_over_bottom

    attrs["rrs_method"] = cast_result.rrs_method or ""
    attrs["rrs_source"] = cast_result.rrs_source or ""
    attrs["shadow_correction_note"] = cast_result.shadow_correction_note or ""
    attrs["bottom_note"] = cast_result.bottom_note or ""

    if ds is not None:
        for key, missing in (("chl_flag", float("nan")), ("qc_flag", -1)):
            value = ds.attrs.get(key)
            attrs[key] = value if value is not None else missing
        attrs["shallow"] = int(bool(ds.attrs.get("shallow", False)))

    # Prefer the position actually used for shadow correction (may come from a
    # PositionOverride or a GPS file via process_deployment(), and so can
    # differ from ds.attrs) over the raw ds.attrs value from info.cops.dat.
    longitude = cast_result.resolved_longitude
    latitude = cast_result.resolved_latitude
    if longitude is None and ds is not None:
        longitude = ds.attrs.get("longitude")
    if latitude is None and ds is not None:
        latitude = ds.attrs.get("latitude")
    attrs["longitude"] = longitude if longitude is not None else float("nan")
    attrs["latitude"] = latitude if latitude is not None else float("nan")

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def write_cast_result(cast_result: CastResult, path: str | Path, ds: xr.Dataset | None = None) -> None:
    """Write ``cast_result`` (see :func:`cast_result_to_dataset`) to a NetCDF file at ``path``."""
    cast_result_to_dataset(cast_result, ds=ds).to_netcdf(Path(path), engine="netcdf4")


def write_deployment_result(
    result: DeploymentProcessingResult,
    directory: str | Path,
    datasets: dict[str, xr.Dataset] | None = None,
) -> dict[str, Path]:
    """Write every cast in ``result.cast_results`` to its own NetCDF file in ``directory``.

    ``directory`` is created if it doesn't exist. Each file is named
    ``<cast file stem>.nc`` (e.g. ``hudsonbay_CAST_001_..._URC.csv`` ->
    ``hudsonbay_CAST_001_..._URC.nc``). ``datasets``, if given -- typically
    :attr:`~pycops.io.discovery.DeploymentCastsResult.datasets` from the same
    :func:`~pycops.io.discovery.read_deployment_casts` call that fed
    :func:`~pycops.processing.deployment.process_deployment` -- supplies the
    original cast per file for the richer output :func:`write_cast_result`
    can produce; a cast missing from it just gets no ``ds`` passed. Returns
    the cast file name -> written path for every file actually written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for file, cast_result in result.cast_results.items():
        path = directory / f"{Path(file).stem}.nc"
        write_cast_result(cast_result, path, ds=(datasets or {}).get(file))
        written[file] = path

    return written
