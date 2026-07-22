"""Discover and read all raw casts in a C-OPS deployment folder.

Ties ``init.cops.dat``, ``info.cops.dat``, ``select.cops.dat`` and the raw
cast files sitting in the same ``COPS*/`` folder into one per-station view.
Port of the file-discovery half of ``process.cops.R``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import xarray as xr

from pycops.io.config import CastInfo, read_info_cops, read_init_cops
from pycops.io.raw import read_cast

# select.cops.dat flag meanings, per process.cops.R's kept.cast/kept.bioS logic.
FLAG_REJECTED = 0
FLAG_NORMAL = 1
FLAG_BIOSHADE = 2
FLAG_UNDER_ICE = 3
_KEPT_FLAGS = (FLAG_NORMAL, FLAG_BIOSHADE, FLAG_UNDER_ICE)

_DEFAULT_METHOD = "Rrs.0p.linear"


@dataclass(frozen=True)
class CastSelection:
    """One row of ``select.cops.dat``: a cast's QC flag and retained Rrs method."""

    file: str
    flag: int
    method: str
    extra: str  # select.cops.dat's 4th field: "1" means SHALLOW (see .shallow)

    @property
    def shallow(self) -> bool:
        """Whether this cast is flagged ``SHALLOW`` -- ``process.cops.R``'s
        ``select.tab[,4] == "1"`` ("Shallow water. Profile finished just
        above the bottom"), gating :mod:`pycops.processing.bottom`.
        """
        return self.extra.strip() == "1"


def read_select_cops(path: str | Path) -> list[CastSelection]:
    """Parse a ``select.cops.dat`` file into one :class:`CastSelection` per cast."""
    path = Path(path)
    selections: list[CastSelection] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [x.strip() for x in line.split(";")]
            fields += [""] * (4 - len(fields))
            selections.append(CastSelection(file=fields[0], flag=int(fields[1]), method=fields[2], extra=fields[3]))

    return selections


def update_cast_selection(path: str | Path, file: str, flag: int, method: str, shallow: bool = False) -> None:
    """Write one ``select.cops.dat`` row (QC flag / Rrs method / SHALLOW), replacing it in place.

    Unlike :func:`pycops.io.config.update_time_window` (which touches one field within a longer,
    heavily-commented row), ``select.cops.dat`` rows are just these four fields with no comment
    header in any real file seen so far -- so this replaces the *whole* row for ``file`` (creating
    it if missing) rather than patching a single field. Still a surgical, line-level edit: every
    other row is left byte-for-byte untouched, including its own line terminator -- real
    deployments have been seen with both CRLF- and LF-only ``select.cops.dat`` files (even across
    sibling stations of the same project), so this preserves whichever a given file already uses
    rather than imposing one convention.
    """
    path = Path(path)
    extra = "1" if shallow else "NA"
    new_row = f"{file};{flag};{method};{extra}"

    if not path.exists():
        path.write_text(new_row + "\n")
        return

    with path.open(newline="") as f:
        text = f.read()
    lines = text.splitlines(keepends=True)

    found = False
    for i, line in enumerate(lines):
        content = line.splitlines()[0] if line else line
        terminator = line[len(content) :]
        stripped = content.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if content.split(";", 1)[0].strip() != file:
            continue
        lines[i] = new_row + terminator
        found = True
        break

    if not found:
        terminator = "\n"
        for line in reversed(lines):
            if line.endswith("\r\n"):
                terminator = "\r\n"
                break
            if line.endswith("\n"):
                terminator = "\n"
                break
        if lines and not lines[-1].endswith(("\n", "\r\n", "\r")):
            lines[-1] += terminator
        lines.append(new_row + terminator)

    with path.open("w", newline="") as f:
        f.write("".join(lines))


@dataclass(frozen=True)
class CastRecord:
    """One cast in a deployment: its raw file path plus info/select metadata."""

    path: Path
    info: CastInfo
    selection: CastSelection

    @property
    def kept(self) -> bool:
        return self.selection.flag in _KEPT_FLAGS


@dataclass(frozen=True)
class Deployment:
    """One ``COPS*/`` deployment folder: processing params plus its cast records."""

    directory: Path
    init: dict[str, object]
    casts: list[CastRecord]

    def kept_casts(self) -> list[CastRecord]:
        return [c for c in self.casts if c.kept]


def find_deployment_folders(parent: str | Path) -> list[Path]:
    """Every directory under ``parent`` (including ``parent`` itself) holding an
    ``init.cops.dat`` -- i.e. every processable deployment folder, found recursively so a parent
    like ``L2/`` (with one ``YYYYMMDD_StationXXX/cops/`` per station) works directly, for batch
    reprocessing across many stations at once."""
    parent = Path(parent)
    return sorted({p.parent for p in parent.rglob("init.cops.dat")})


def discover_deployment(directory: str | Path) -> Deployment:
    """Parse a ``COPS*/`` folder's config files into a :class:`Deployment`.

    If ``select.cops.dat`` is absent, or is missing a row for a given cast,
    that cast is treated as a kept, normal-flag profile -- mirroring the R
    package, which auto-generates that default the first time a deployment
    is processed.
    """
    directory = Path(directory)
    init = read_init_cops(directory / "init.cops.dat")
    info_entries = read_info_cops(directory / "info.cops.dat")

    select_path = directory / "select.cops.dat"
    selections = {s.file: s for s in read_select_cops(select_path)} if select_path.exists() else {}

    casts = [
        CastRecord(
            path=directory / info.file,
            info=info,
            selection=selections.get(
                info.file, CastSelection(file=info.file, flag=FLAG_NORMAL, method=_DEFAULT_METHOD, extra="")
            ),
        )
        for info in info_entries
    ]

    return Deployment(directory=directory, init=init, casts=casts)


@dataclass(frozen=True)
class CastReadFailure:
    """One cast that failed to parse/read; captured instead of aborting the whole batch."""

    file: str
    path: Path
    error: str  # f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class DeploymentCastsResult:
    """Return value of :func:`read_deployment_casts`: successes plus any per-cast failures."""

    datasets: dict[str, xr.Dataset]
    failures: list[CastReadFailure]


def read_deployment_casts(deployment: Deployment, only_kept: bool = True) -> DeploymentCastsResult:
    """Read every (kept) cast of a :class:`Deployment` into an ``xarray.Dataset``.

    Each dataset is annotated with its ``info.cops.dat`` position, chlorophyll/
    absorption-source flag, and ``select.cops.dat`` QC flag/method as attrs,
    keyed by cast file name. A cast that fails to read (e.g. an unparseable
    file name, given how much raw file-naming conventions vary across ~15
    years of deployments) doesn't abort the rest of the batch: it's recorded
    in the result's ``failures`` list with a warning instead.
    """
    instruments = tuple(deployment.init["instruments.optics"])
    n_fields = int(deployment.init["number.of.fields.before.date"])

    records = deployment.kept_casts() if only_kept else deployment.casts
    datasets: dict[str, xr.Dataset] = {}
    failures: list[CastReadFailure] = []
    for record in records:
        try:
            ds = read_cast(record.path, instruments=instruments, number_of_fields_before_date=n_fields)
            ds.attrs.update(
                longitude=record.info.longitude,
                latitude=record.info.latitude,
                chl_flag=record.info.chl_flag,
                qc_flag=record.selection.flag,
                rrs_method=record.selection.method,
                shallow=record.selection.shallow,
                time_window=record.info.time_window,
            )
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: isolate one bad cast from the rest
            error = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"{deployment.directory.name}: failed to read cast {record.info.file!r} ({error}); "
                f"skipping and continuing",
                stacklevel=2,
            )
            failures.append(CastReadFailure(file=record.info.file, path=record.path, error=error))
            continue
        datasets[record.info.file] = ds

    return DeploymentCastsResult(datasets=datasets, failures=failures)
