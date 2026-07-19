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
