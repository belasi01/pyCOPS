"""Process every kept cast in a ``COPS*/`` deployment folder end-to-end.

Ties :mod:`pycops.io.discovery`'s file-discovery layer to
:func:`pycops.processing.process_cast.process_cast`'s per-cast pipeline:
reads every kept cast, processes any BioShade cast(s) (``select.cops.dat``
flag ``2``) first so their measured diffuse/total split can be passed to
every other cast's shadow correction, then runs ``process_cast`` on each
remaining cast with the matching ``absorption.cops.dat`` row when its
``chl`` flag is ``0``. Equivalent to one full iteration of ``process.cops.R``'s
outer loop over one station -- still missing the R package's PDF diagnostics
and ``generate.cops.DB()`` aggregation step.

Also resolves each cast's position when ``info.cops.dat`` doesn't have it: a
``GPS_*.tsv`` file in the deployment folder (see
:mod:`pycops.processing.position`) is used automatically, and an explicit
per-file ``position_overrides`` mapping takes priority over both -- for a
cast whose own GPS failed in the field, a real recurring situation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import xarray as xr

from pycops.io.config import absorption_for_cast, read_absorption_cops
from pycops.io.discovery import (
    CastReadFailure,
    Deployment,
    FLAG_BIOSHADE,
    discover_deployment,
    read_deployment_casts,
    read_one_cast,
)
from pycops.io.exclusions import read_wavelength_exclusions
from pycops.processing.bioshade import BioShadeResult, process_bioshade
from pycops.processing.position import PositionOverride, find_gps_file, position_from_gps, read_gps_file
from pycops.processing.process_cast import CastResult, process_cast


@dataclass(frozen=True)
class CastProcessingFailure:
    """One cast that read fine but failed during ``process_bioshade``/``process_cast``."""

    file: str
    error: str  # f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class DeploymentProcessingResult:
    """Result of processing every kept cast in a deployment."""

    cast_results: dict[str, CastResult]  # by cast file name, profile casts only
    bioshade_results: dict[str, BioShadeResult]  # by cast file name, BioShade casts
    bioshade_used: BioShadeResult | None  # the one passed to every cast_results entry, if any
    read_failures: list[CastReadFailure] = field(default_factory=list)
    processing_failures: list[CastProcessingFailure] = field(default_factory=list)


@dataclass(frozen=True)
class ReprocessedCast:
    """Result of :func:`reprocess_single_cast`: the fit plus the annotated dataset it was fit
    from -- a caller writing this back out (see :func:`pycops.io.netcdf.write_cast_result`) needs
    ``ds`` too, since that's what carries the real ``time`` coordinate and the
    ``chl_flag``/``qc_flag``/``shallow``/position attrs the ``.nc`` file's own attrs come from."""

    result: CastResult
    ds: xr.Dataset


def _load_absorption_table(directory: Path) -> pd.DataFrame | None:
    path = directory / "absorption.cops.dat"
    return read_absorption_cops(path) if path.exists() else None


def _load_wavelength_exclusions(directory: Path) -> dict[str, list[float]]:
    return read_wavelength_exclusions(directory / "rrs_wavelength_exclusions.cops.dat")


def _load_gps_table(directory: Path) -> pd.DataFrame | None:
    gps_path = find_gps_file(directory)
    if gps_path is None:
        return None
    try:
        return read_gps_file(gps_path)
    except Exception as exc:  # noqa: BLE001 -- a bad GPS file shouldn't block the whole deployment
        warnings.warn(f"{directory.name}: failed to read GPS file {gps_path.name!r} ({exc}); ignoring", stacklevel=2)
        return None


def _resolve_position(
    ds, file: str, position_overrides: dict[str, PositionOverride] | None, gps_table: pd.DataFrame | None
) -> PositionOverride | None:
    override = (position_overrides or {}).get(file)
    if override is not None and (override.longitude is not None or override.latitude is not None):
        return override  # an explicit manual position wins outright

    if gps_table is not None and (ds.attrs.get("longitude") is None or ds.attrs.get("latitude") is None):
        found = position_from_gps(gps_table, ds["time"].values)
        if found is not None:
            longitude, latitude = found
            utc_time = override.utc_time if override is not None else None
            return PositionOverride(longitude=longitude, latitude=latitude, utc_time=utc_time)

    return override  # None, or a utc_time-only override with no position fix available


def _process_kept_cast(
    ds,
    init: dict[str, object],
    file: str,
    chl_flag: float | None,
    absorption_table: pd.DataFrame | None,
    gps_table: pd.DataFrame | None,
    bioshade_used: BioShadeResult | None,
    position_overrides: dict[str, PositionOverride] | None,
    excluded_wavelengths: list[float] | None = None,
) -> CastResult:
    """The per-cast body shared by :func:`process_deployment`'s loop and
    :func:`reprocess_single_cast`, so both always process one cast exactly the same way."""
    absorption_waves = absorption_values = None
    if chl_flag == 0 and absorption_table is not None:
        try:
            absorption_waves, absorption_values = absorption_for_cast(absorption_table, file)
        except KeyError:
            pass  # process_cast reports the missing row via shadow_correction_note

    position_override = _resolve_position(ds, file, position_overrides, gps_table)

    return process_cast(
        ds,
        init,
        absorption_waves=absorption_waves,
        absorption_values=absorption_values,
        bioshade=bioshade_used,
        position_override=position_override,
        excluded_wavelengths=excluded_wavelengths,
    )


def _find_bioshade_result(directory: Path, deployment: Deployment) -> BioShadeResult | None:
    """The first BioShade cast (``select.cops.dat`` flag ``2``) that processes cleanly, if any.

    Only used by :func:`reprocess_single_cast`, which -- unlike :func:`process_deployment`'s own
    bioshade loop -- doesn't need to track every BioShade cast's own result/failures, just
    *a* usable diffuse/direct split to pass into the one cast actually being reprocessed.
    """
    for record in deployment.kept_casts():
        if record.selection.flag != FLAG_BIOSHADE:
            continue
        try:
            ds = read_one_cast(record, deployment.init)
            return process_bioshade(ds, deployment.init)
        except Exception as exc:  # noqa: BLE001 -- fall through to the next BioShade cast, if any
            warnings.warn(
                f"{directory.name}: failed to process BioShade cast {record.info.file!r} "
                f"({type(exc).__name__}: {exc})",
                stacklevel=2,
            )
    return None


def reprocess_single_cast(
    directory: str | Path,
    file: str,
    position_overrides: dict[str, PositionOverride] | None = None,
) -> ReprocessedCast:
    """Reprocess exactly one cast in a deployment folder.

    Re-parses ``info.cops.dat``/``select.cops.dat`` fresh (:func:`~pycops.io.discovery.discover_deployment`,
    cheap -- just the small text config files) so a just-saved per-cast override or position edit
    takes effect, then reads only ``file`` itself (not every sibling cast's raw file, unlike a
    full :func:`process_deployment` run -- meant for an interactive "adjust one cast's parameters
    and reprocess" workflow, where re-reading a whole multi-cast station each time would be slow).

    Still resolves the deployment's BioShade cast (if any) and the absorption/GPS tables exactly
    like :func:`process_deployment` does, since shadow correction needs them -- this calls the
    same per-cast body internally, so the result matches exactly what a full reprocess would give
    for this one cast. Raises ``ValueError`` if ``file`` isn't a row in this deployment's
    ``info.cops.dat``; propagates any error from reading/processing the cast itself (unlike
    ``process_deployment``, which isolates per-cast failures -- a caller reprocessing one cast
    interactively wants to know immediately if it failed).

    Returns both the fit and the annotated dataset it was fit from (see
    :class:`ReprocessedCast`) -- a caller writing the result back to the cast's ``.nc`` file
    (:func:`pycops.io.netcdf.write_cast_result`) needs the *annotated* ``ds`` (real ``time``
    coordinate, ``chl_flag``/``qc_flag``/``shallow``/position attrs), not a bare ``read_cast()``.
    """
    directory = Path(directory)
    deployment = discover_deployment(directory)
    record = next((r for r in deployment.casts if r.info.file == file), None)
    if record is None:
        raise ValueError(f"{file!r} not found in {directory}'s info.cops.dat")

    absorption_table = _load_absorption_table(directory)
    gps_table = _load_gps_table(directory)
    bioshade_used = _find_bioshade_result(directory, deployment)
    excluded_wavelengths = _load_wavelength_exclusions(directory).get(file)

    ds = read_one_cast(record, deployment.init)
    result = _process_kept_cast(
        ds,
        deployment.init,
        file,
        record.info.chl_flag,
        absorption_table,
        gps_table,
        bioshade_used,
        position_overrides,
        excluded_wavelengths,
    )
    return ReprocessedCast(result=result, ds=ds)


def process_deployment(
    directory: str | Path,
    position_overrides: dict[str, PositionOverride] | None = None,
) -> DeploymentProcessingResult:
    """Read and process every kept cast in a deployment folder.

    ``directory`` is a ``COPS*/`` folder holding ``init.cops.dat``,
    ``info.cops.dat``, ``select.cops.dat``, and the raw cast files. Casts
    flagged ``2`` (BioShade) in ``select.cops.dat`` are processed with
    :func:`~pycops.processing.bioshade.process_bioshade` first; the first one
    that processes cleanly is passed as every other kept cast's ``bioshade``
    argument to :func:`~pycops.processing.process_cast.process_cast` (there's
    no documented tie-break in the R package for a folder with more than one
    usable BioShade cast). A cast whose ``info.cops.dat`` ``chl`` flag is
    ``0`` gets its absorption spectrum looked up from ``absorption.cops.dat``
    by file name; if that file or row is missing, ``process_cast`` itself
    reports why shadow correction was skipped for that cast (see
    ``CastResult.shadow_correction_note``) rather than this function raising.
    A cast with a row in ``rrs_wavelength_exclusions.cops.dat`` (see
    :mod:`pycops.io.exclusions`) gets those wavelengths NaN'd out of its final
    Rrs -- a pycops-only, post-fit QC override with no R equivalent.

    A cast that fails to read is recorded in ``read_failures`` (see
    :func:`~pycops.io.discovery.read_deployment_casts`); one that reads but
    raises inside ``process_bioshade``/``process_cast`` is recorded in
    ``processing_failures`` instead of aborting the rest of the deployment.

    Position resolution, per cast, in priority order: (1)
    ``position_overrides[file]`` if it supplies a longitude/latitude
    (``dict`` keyed by cast file name -- for a cast whose own GPS is known to
    be wrong or missing entirely, supplied directly by the researcher); (2)
    ``info.cops.dat``'s longitude/latitude, if not ``NA``; (3) a
    ``GPS_*.tsv``/``.csv`` file in ``directory``, if one exists and its own
    clock overlaps the cast's recorded time (see
    :mod:`pycops.processing.position`); (4) otherwise unavailable, and
    ``CastResult.shadow_correction_note`` says so. A ``utc_time`` override
    applies independently of position (e.g. the cast's own clock, not just
    its GPS fix, is known to be wrong).
    """
    directory = Path(directory)
    deployment: Deployment = discover_deployment(directory)
    read_result = read_deployment_casts(deployment)
    absorption_table = _load_absorption_table(directory)
    gps_table = _load_gps_table(directory)
    wavelength_exclusions = _load_wavelength_exclusions(directory)
    chl_flag_by_file = {record.info.file: record.info.chl_flag for record in deployment.kept_casts()}

    bioshade_files = [
        record.info.file for record in deployment.kept_casts() if record.selection.flag == FLAG_BIOSHADE
    ]

    processing_failures: list[CastProcessingFailure] = []
    bioshade_results: dict[str, BioShadeResult] = {}
    for file in bioshade_files:
        ds = read_result.datasets.get(file)
        if ds is None:
            continue  # already recorded in read_result.failures
        try:
            bioshade_results[file] = process_bioshade(ds, deployment.init)
        except Exception as exc:  # noqa: BLE001 -- isolate one bad BioShade cast from the rest
            error = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"{directory.name}: failed to process BioShade cast {file!r} ({error})", stacklevel=2)
            processing_failures.append(CastProcessingFailure(file=file, error=error))

    bioshade_used = next(iter(bioshade_results.values()), None)

    cast_results: dict[str, CastResult] = {}
    for file, ds in read_result.datasets.items():
        if file in bioshade_files:
            continue
        try:
            cast_results[file] = _process_kept_cast(
                ds,
                deployment.init,
                file,
                chl_flag_by_file.get(file),
                absorption_table,
                gps_table,
                bioshade_used,
                position_overrides,
                wavelength_exclusions.get(file),
            )
        except Exception as exc:  # noqa: BLE001 -- isolate one bad cast from the rest
            error = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"{directory.name}: failed to process cast {file!r} ({error})", stacklevel=2)
            processing_failures.append(CastProcessingFailure(file=file, error=error))

    return DeploymentProcessingResult(
        cast_results=cast_results,
        bioshade_results=bioshade_results,
        bioshade_used=bioshade_used,
        read_failures=read_result.failures,
        processing_failures=processing_failures,
    )
