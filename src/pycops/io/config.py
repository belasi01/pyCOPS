"""Read the ``init.cops.dat`` and ``info.cops.dat`` configuration files.

These sit alongside raw cast files in every ``COPS*/`` deployment folder.
``init.cops.dat`` holds processing parameters for the whole deployment;
``info.cops.dat`` is a table with one row per cast file giving its
position and any per-cast overrides. Ports ``read.init.R`` from the
``Cops`` R package.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# init.cops.dat parameters that are vectors indexed by instruments.optics.
_PER_INSTRUMENT_KEYS = (
    "tiltmax.optics",
    "time.interval.for.smoothing.optics",
    "depth.interval.for.smoothing.optics",
    "sub.surface.removed.layer.optics",
    "delta.capteur.optics",
    "radius.instrument.optics",
    "linear.fit.Rsquared.threshold.optics",
    "linear.fit.max.delta.depth.optics",
)

# Fallback defaults for parameters added to the R package after many
# init.cops.dat files were already written (read.init.R warns and injects
# these same values -- and rewrites the file -- the first time it hits an
# older file missing them). Ed0 gets NaN: no linear-fit threshold applies at
# the surface.
_PER_INSTRUMENT_DEFAULTS = {
    "linear.fit.Rsquared.threshold.optics": {"Ed0": float("nan"), "EdZ": 0.5, "LuZ": 0.6, "EuZ": 0.6},
    "linear.fit.max.delta.depth.optics": {"Ed0": float("nan"), "EdZ": 3.0, "LuZ": 2.5, "EuZ": 2.5},
}
_SCALAR_DEFAULTS = {
    "bandwidth": 10.0,
    "windspeed_ms": 4.0,
}

def _to_float(value: str) -> float:
    # "NA" is R's missing-value sentinel, e.g. for a linear-fit threshold that
    # doesn't apply to the surface (Ed0) instrument.
    return float("nan") if value.strip().upper() == "NA" else float(value)


_CASTERS = {
    "numeric": lambda values: [_to_float(v) for v in values],
    "logical": lambda values: [v.strip().upper() == "TRUE" for v in values],
    "character": lambda values: [v.strip() for v in values],
}


def read_init_cops(path: str | Path) -> dict[str, object]:
    """Parse an ``init.cops.dat`` file into a dict of processing parameters.

    Scalar numeric/character/logical values are unwrapped from their
    single-element list; the parameters listed in ``_PER_INSTRUMENT_KEYS``
    are returned as ``{instrument: value}`` dicts keyed by
    ``instruments.optics``.
    """
    path = Path(path)
    raw: dict[str, list[object]] = {}

    with path.open() as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            name, kind, value = (part.strip() for part in line.split(";", 2))
            values = _CASTERS[kind]([v for v in value.split(",")])
            raw[name] = values

    instruments = raw.get("instruments.optics", [])

    params: dict[str, object] = {}
    for name, values in raw.items():
        if name in _PER_INSTRUMENT_KEYS:
            params[name] = dict(zip(instruments, values))
        else:
            params[name] = values[0] if len(values) == 1 else values

    for name, defaults_by_instrument in _PER_INSTRUMENT_DEFAULTS.items():
        if name not in params:
            warnings.warn(
                f"{path.name} has no {name!r}; using default values {defaults_by_instrument}",
                stacklevel=2,
            )
            params[name] = {instr: defaults_by_instrument[instr] for instr in instruments}

    for name, default in _SCALAR_DEFAULTS.items():
        if name not in params:
            warnings.warn(f"{path.name} has no {name!r}; defaulting to {default}", stacklevel=2)
            params[name] = default

    return params


@dataclass(frozen=True)
class CastInfo:
    """One row of ``info.cops.dat``: a cast file's position and overrides."""

    file: str
    longitude: float | None
    latitude: float | None
    chl_flag: float | None
    time_window: tuple[float, float] | None
    sub_surface_removed_layer: list[float] | None
    tiltmax: list[float] | None
    depth_interval_for_smoothing: list[float] | None
    dark_files: list[str]


def _override(field: str) -> list[float] | None:
    field = field.strip()
    if field in ("", "x"):
        return None
    return [float(v) for v in field.split(",")]


def _optional_float(field: str) -> float | None:
    field = field.strip()
    return None if field.upper() == "NA" else float(field)


def read_info_cops(path: str | Path) -> list[CastInfo]:
    """Parse an ``info.cops.dat`` file into one :class:`CastInfo` per cast."""
    path = Path(path)
    entries: list[CastInfo] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [f.strip() for f in line.split(";")]
            fields += [""] * (12 - len(fields))

            time_window = _override(fields[4])

            entries.append(
                CastInfo(
                    file=fields[0],
                    longitude=_optional_float(fields[1]),
                    latitude=_optional_float(fields[2]),
                    chl_flag=_optional_float(fields[3]),
                    time_window=tuple(time_window) if time_window else None,
                    sub_surface_removed_layer=_override(fields[5]),
                    tiltmax=_override(fields[6]),
                    depth_interval_for_smoothing=_override(fields[7]),
                    dark_files=[f for f in fields[8:12] if f],
                )
            )

    return entries


def info_cops_to_frame(entries: list[CastInfo]) -> pd.DataFrame:
    """Convenience wrapper: :func:`read_info_cops` results as a flat DataFrame."""
    return pd.DataFrame([vars(e) for e in entries])


# Verbatim header comment block from a real cops.go()-generated info.cops.dat
# (local_data/AlgaeWISE/20220705_StationPME6/COPS/info.cops.dat) -- so a file
# freshly created by update_time_window() looks like one the R package itself
# would produce, joined with "\r\n" at write time to match real files' line
# endings (see update_time_window).
_INFO_COPS_DAT_HEADER = [
    "#" * 135,
    "#                           this file is a table with a maximum of 12 fields separated by \";\"                                         #",
    "#" * 135,
    "# the 8 first fields are mandatory                                                                                                    #",
    "#" * 135,
    "#-fields-- 1 2 3 4 ------#                                                                                                            #",
    "#-------------------------                                                                                                            #",
    "# field #1 : file name                                                                                                                #",
    "# field #2 : longitude (decimal degs)                                                                                                 #",
    "# field #3 : latitude (decimal degs)                                                                                                  #",
    "# field #4 : a positive value, 0, 999, or NA                                                                                          #",
    "#           if > 0, it is the chlorophyll concentration, which is used to calculate absorption at each wavelength (case 1 waters model)#",
    "#           if = 0, the absorption coefficients for all wavelengths must be found in a file called absorption.cops.dat                #",
    "#           if 999, the absorption coefficients are derived from Kd near the sea surface",
    "#           if NA, no shadow correction                                                                                               #",
    "#" * 135,
    "#-fields-- 5 6 7 8 ------#                                                                                                            #",
    "#-------------------------                                                                                                            #",
    "# NB : fields 5 6 7 8 9 10 are defined for the whole cast in file init.cops.dat                                                       #",
    "#      if no peculiar value is needed for an experiment, just put a \"x\" in the field, the value in init.cops.dat is kept              #",
    "#-------------------------                                                                                                            #",
    "# field #5 : time.window, 2 numeric values separated by \",\"; no space                                                                 #",
    "# field #6 : sub.surface.removed.layer, N numeric separated by \",\"; no space (N is the number of optics instruments - respect order)   #",
    "# field #7 : tiltmax, N numeric separated by \",\"; no space (N is the number of optics instruments - respect order)                     #",
    "# field #8 : depth.interval.for.smoothing, N numeric separated by \",\"; no space (N is the number of optics instruments - respect order)#",
    "# field #9 : linear.fit.Rsquared.threshold, N numeric separated by \",\"; no space (N is the number of optics instruments - respect order)#",
    "# field #10 : linear.fit.max.delta.depth,  N numeric separated by \",\"; no space (N is the number of optics instruments - respect order)#",
    "#",
    "#",
    "#" * 135,
    "# 4 optional fields ( #11 #12 #13 and #14 ) : names of files containing dark measures                                                  #",
    "#" * 135,
    "# examples                                                                                                                            #",
    "#110923_0237_001_data.txt;7.1;43.9;0.1;0,90;0.1,0.05,0.1;10,10,5;10,12,15;x;x;110923_0237_004_data.txt                                    #",
    "#110923_0237_001_data.txt;7.1;43.9;0;x;0.1,0.05,0.1;x;x;x;x;110923_0237_004_data.txt;110923_0237_005_data.txt                             #",
    "#110923_0237_001_data.txt;7.1;43.9;0.2;x;x;10,10,5;x;x;x                                                                           #",
    "#" * 133 + " #",
]


def _format_time_window(time_window: tuple[float, float]) -> str:
    start, end = time_window
    return f"{start:g},{end:g}"


def update_time_window(path: str | Path, file: str, time_window: tuple[float, float]) -> None:
    """Write a per-cast ``time.window`` override into ``info.cops.dat``, touching only that field.

    A surgical line-level edit rather than a full read-into-:class:`CastInfo`-then-rewrite-
    everything round trip, so it never disturbs a row/field/comment it doesn't own -- real files
    carry a large hand-written comment header, and other fields (e.g. ``dark_files``) round-trip
    imperfectly through :func:`read_info_cops` today. Real ``info.cops.dat`` files are
    CRLF-terminated; this preserves whatever line ending each existing line already has (untouched
    lines round-trip byte-for-byte) rather than normalizing the whole file to ``\\n``.

    If ``file`` already has a row, only its field 5 (``time.window``) is replaced -- padding the
    row out to 5 fields with ``"x"`` first if it was shorter. If the file exists but has no row for
    ``file``, one is appended (matching a real minimal row's shape: lon/lat/chl ``NA``, other
    overrides ``"x"``). If ``path`` doesn't exist at all, it's created with the same header comment
    block a real ``cops.go()``-generated file has, plus that one row.
    """
    path = Path(path)
    formatted = _format_time_window(time_window)

    if not path.exists():
        row = f"{file};NA;NA;NA;{formatted};x;x;x;x;x"
        content = "\r\n".join([*_INFO_COPS_DAT_HEADER, row]) + "\r\n"
        path.write_text(content, newline="")
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
        fields = content.split(";")
        if fields[0].strip() != file:
            continue
        if len(fields) <= 4:
            fields += ["x"] * (5 - len(fields))
        fields[4] = formatted
        lines[i] = ";".join(fields) + terminator
        found = True
        break

    if not found:
        terminator = "\r\n"
        for line in reversed(lines):
            if line.endswith("\r\n"):
                terminator = "\r\n"
                break
            if line.endswith("\n"):
                terminator = "\n"
                break
        if lines and not lines[-1].endswith(("\n", "\r\n", "\r")):
            lines[-1] += terminator
        lines.append(f"{file};NA;NA;NA;{formatted};x;x;x;x;x{terminator}")

    with path.open("w", newline="") as f:
        f.write("".join(lines))


def read_absorption_cops(path: str | Path) -> pd.DataFrame:
    """Parse an ``absorption.cops.dat`` file: one row of absorption a(lambda) per cast.

    Used for shadow correction when ``info.cops.dat``'s ``chl`` field is ``0``
    (see :func:`pycops.processing.shadow.resolve_absorption`). Returns a
    DataFrame indexed by cast file name, with wavelength (nm) column labels --
    port of the ``read.table(..., row.names = 1)`` call in ``process.cops.R``.
    """
    return pd.read_csv(Path(path), sep=";", index_col=0)


def absorption_for_cast(table: pd.DataFrame, filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Absorption waves/values for one cast from a table loaded by :func:`read_absorption_cops`."""
    waves = table.columns.to_numpy(dtype=float)
    values = table.loc[filename].to_numpy(dtype=float)
    return waves, values
