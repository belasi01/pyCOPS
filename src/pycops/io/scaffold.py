"""Scaffold a new ``L2/YYYYMMDD_Station<ID>/cops/`` station folder from ``L1`` raw cast files.

The first step of Simon's real workflow, before any of the interactive cleaning/editing in
:mod:`pycops.ui.clean_app`: at the end of a station or a day at sea, ``L1`` raw exports get
manually placed in a read-only-protected folder and never touched again; for each station, a new
``L2`` folder gets created and the relevant cast files (plus that day's GPS file) copied in. This
module only ever *reads* from the ``L1`` folder and *creates*/*copies into* the ``L2`` folder --
it never modifies or deletes anything in ``L1``, preserving the read-only-raw-data guarantee.

Folder naming (per Simon, 2026-07-19): the date comes from the selected cast files' own names
(see :func:`pycops.io.raw.parse_cast_filename`), not typed by hand; the only thing the researcher
enters is the station ID. The child folder is always named ``cops`` (lowercase, no operator
suffix) -- a newly-standardized convention, not necessarily matching every legacy folder name
already on sabre (e.g. ``COPS_FJSaucier``, ``COPS_Kildir``).
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pycops.io.raw import parse_cast_filename

_CAST_GLOBS = ("*_URC.tsv", "*_URC.csv")


def discover_l1_casts(l1_directory: str | Path) -> list[Path]:
    """Cast files in an ``L1`` folder, sorted by their own recorded date/time.

    Same glob/parse/skip-unparseable pattern as
    :func:`pycops.ui.clean_app._discover_casts` (an ``L1`` folder has no config files to
    cross-check against, only the raw cast files themselves).
    """
    directory = Path(l1_directory)
    candidates = [p for pattern in _CAST_GLOBS for p in directory.glob(pattern)]
    parsed: list[tuple[Path, object]] = []
    for path in sorted(candidates):
        try:
            info = parse_cast_filename(path)
        except ValueError as exc:
            warnings.warn(f"skipping {path.name}: {exc}", stacklevel=2)
            continue
        parsed.append((path, info.date))
    parsed.sort(key=lambda item: item[1])
    return [path for path, _ in parsed]


def find_gps_files_for_date(l1_directory: str | Path, day: date) -> list[Path]:
    """``GPS_<YYMMDD>.*`` files in ``l1_directory`` matching ``day`` (uProfile's GPS-file naming)."""
    directory = Path(l1_directory)
    stem = f"GPS_{day.strftime('%y%m%d')}"
    return sorted(directory.glob(f"{stem}*"))


def validate_station_id(station_id: str) -> str:
    """Strip and sanity-check a station ID before it's used as a folder-name component.

    Rejects anything that looks like a path rather than a bare identifier (e.g. containing ``/``)
    -- it gets interpolated directly into a filesystem path.
    """
    station_id = station_id.strip()
    if not station_id:
        raise ValueError("station_id must not be blank")
    if "/" in station_id or "\\" in station_id or station_id in (".", ".."):
        raise ValueError(f"station_id {station_id!r} looks like a path, not a bare station identifier")
    return station_id


@dataclass(frozen=True)
class ScaffoldResult:
    """What :func:`scaffold_station` actually did."""

    destination: Path
    date_used: date
    copied_casts: list[str] = field(default_factory=list)
    copied_gps_files: list[str] = field(default_factory=list)
    copied_init: bool = False
    skipped_existing: list[str] = field(default_factory=list)  # already present, not overwritten


def scaffold_station(
    l1_directory: str | Path,
    l2_parent: str | Path,
    station_id: str,
    cast_files: list[str],
    init_cops_dat_template: str | Path | None = None,
    overwrite: bool = False,
) -> ScaffoldResult:
    """Create ``<l2_parent>/<YYYYMMDD>_Station<station_id>/cops/`` and populate it.

    ``cast_files`` are file names (not full paths) of cast files already present in
    ``l1_directory`` -- the researcher's manual selection of which of that day's casts belong to
    this station (an ``L1`` day folder can hold more than one station's casts). The destination
    folder's date comes from the *first* selected cast's own file name; if the others disagree, a
    warning is raised (not an error -- matches this codebase's existing "warn and continue" QC
    conventions) since a mismatched selection is much more likely to be a UI mistake worth
    flagging than something to hard-block on.

    Every ``GPS_<date>.*`` file found in ``l1_directory`` for that date is copied alongside the
    casts (GPS logs cover a whole day, needed for :mod:`pycops.processing.position`).
    ``init_cops_dat_template``, if given, is copied in as this new station's starting
    ``init.cops.dat`` (Simon: this rarely changes per instrument system, so starting from an
    existing one is much less tedious than typing it from scratch).

    Never overwrites an existing destination file unless ``overwrite=True`` -- anything skipped
    for that reason is reported in ``ScaffoldResult.skipped_existing`` rather than silently lost.
    ``L1`` is only ever read from, never modified.
    """
    l1_directory = Path(l1_directory)
    l2_parent = Path(l2_parent)
    station_id = validate_station_id(station_id)
    if not cast_files:
        raise ValueError("no cast files selected")

    dates = [parse_cast_filename(l1_directory / filename).date.date() for filename in cast_files]
    primary_date = dates[0]
    if any(d != primary_date for d in dates):
        warnings.warn(
            f"selected casts span multiple dates ({sorted(set(dates))}); using {primary_date} "
            "(the first selected cast's date) for the station folder name -- double-check the "
            "selection if that's unexpected",
            stacklevel=2,
        )

    station_dir = l2_parent / f"{primary_date.strftime('%Y%m%d')}_Station{station_id}" / "cops"
    station_dir.mkdir(parents=True, exist_ok=True)

    copied_casts: list[str] = []
    skipped: list[str] = []

    def _copy(src: Path, dst_name: str) -> bool:
        dst = station_dir / dst_name
        if dst.exists() and not overwrite:
            skipped.append(dst_name)
            return False
        shutil.copy2(src, dst)
        return True

    for filename in cast_files:
        if _copy(l1_directory / filename, filename):
            copied_casts.append(filename)

    copied_gps: list[str] = []
    for gps_path in find_gps_files_for_date(l1_directory, primary_date):
        if _copy(gps_path, gps_path.name):
            copied_gps.append(gps_path.name)

    copied_init = False
    if init_cops_dat_template is not None:
        copied_init = _copy(Path(init_cops_dat_template), "init.cops.dat")

    return ScaffoldResult(
        destination=station_dir,
        date_used=primary_date,
        copied_casts=copied_casts,
        copied_gps_files=copied_gps,
        copied_init=copied_init,
        skipped_existing=skipped,
    )
