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
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pycops.io.config import absorption_for_cast, read_absorption_cops
from pycops.io.discovery import (
    CastReadFailure,
    Deployment,
    FLAG_BIOSHADE,
    discover_deployment,
    read_deployment_casts,
)
from pycops.processing.bioshade import BioShadeResult, process_bioshade
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


def _load_absorption_table(directory: Path) -> pd.DataFrame | None:
    path = directory / "absorption.cops.dat"
    return read_absorption_cops(path) if path.exists() else None


def process_deployment(directory: str | Path) -> DeploymentProcessingResult:
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

    A cast that fails to read is recorded in ``read_failures`` (see
    :func:`~pycops.io.discovery.read_deployment_casts`); one that reads but
    raises inside ``process_bioshade``/``process_cast`` is recorded in
    ``processing_failures`` instead of aborting the rest of the deployment.
    """
    directory = Path(directory)
    deployment: Deployment = discover_deployment(directory)
    read_result = read_deployment_casts(deployment)
    absorption_table = _load_absorption_table(directory)
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

        absorption_waves = absorption_values = None
        if chl_flag_by_file.get(file) == 0 and absorption_table is not None:
            try:
                absorption_waves, absorption_values = absorption_for_cast(absorption_table, file)
            except KeyError:
                pass  # process_cast reports the missing row via shadow_correction_note

        try:
            cast_results[file] = process_cast(
                ds,
                deployment.init,
                absorption_waves=absorption_waves,
                absorption_values=absorption_values,
                bioshade=bioshade_used,
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
