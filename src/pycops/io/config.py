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

_PER_INSTRUMENT_KEYS_INSTRUMENTS = ("Ed0", "EdZ", "LuZ", "EuZ")

# Starting-point per-instrument values for a brand-new init.cops.dat (see
# default_init_cops_params/write_init_cops below), taken from every real deployment cached in
# local_data/*/*/COPS*/init.cops.dat. radius.instrument.optics is identical (0.035) in all four;
# delta.capteur.optics is near-identical per instrument (Ed0=0, EdZ=-0.05, LuZ/EuZ~0.238) but is a
# physical sensor-offset property -- a caller-facing form should still prompt the researcher to
# confirm it against the instrument's own calibration sheet rather than trust this blindly.
_INIT_COPS_DAT_INSTRUMENT_DEFAULTS: dict[str, dict[str, float]] = {
    "tiltmax.optics": {"Ed0": 10.0, "EdZ": 5.0, "LuZ": 5.0, "EuZ": 5.0},
    "depth.interval.for.smoothing.optics": {"Ed0": 10.0, "EdZ": 5.0, "LuZ": 5.0, "EuZ": 5.0},
    "sub.surface.removed.layer.optics": {"Ed0": 0.0, "EdZ": 0.2, "LuZ": 0.0, "EuZ": 0.0},
    "delta.capteur.optics": {"Ed0": 0.0, "EdZ": -0.05, "LuZ": 0.238, "EuZ": 0.238},
    "radius.instrument.optics": {"Ed0": 0.035, "EdZ": 0.035, "LuZ": 0.035, "EuZ": 0.035},
    # Same values read_init_cops backfills (with a warning) into older files missing these two --
    # kept as one source of truth via _PER_INSTRUMENT_DEFAULTS below. Ed0 gets NaN: no linear-fit
    # threshold applies at the surface.
    "linear.fit.Rsquared.threshold.optics": {"Ed0": float("nan"), "EdZ": 0.5, "LuZ": 0.6, "EuZ": 0.6},
    "linear.fit.max.delta.depth.optics": {"Ed0": float("nan"), "EdZ": 3.0, "LuZ": 2.5, "EuZ": 2.5},
}

# Fallback defaults for parameters added to the R package after many
# init.cops.dat files were already written (read.init.R warns and injects
# these same values -- and rewrites the file -- the first time it hits an
# older file missing them).
_PER_INSTRUMENT_DEFAULTS = {
    name: _INIT_COPS_DAT_INSTRUMENT_DEFAULTS[name]
    for name in ("linear.fit.Rsquared.threshold.optics", "linear.fit.max.delta.depth.optics")
}
_SCALAR_DEFAULTS = {
    "bandwidth": 10.0,
    "windspeed_ms": 4.0,
}

# init.cops.dat parameters that are effectively constant across every real deployment cached in
# local_data/*/*/COPS*/init.cops.dat, regardless of project or instrument system.
_INIT_COPS_DAT_CONSTANTS: dict[str, object] = {
    "verbose": True,
    "indice.water": 1.34,
    "rau.Fresnel": 0.043,
    "win.width": 9.0,
    "win.height": 7.0,
    "format.date": "%m/%d/%Y %H:%M:%S",
    "instruments.others": "NA",
    "time.window": [0.0, 10000.0],
    "depth.discretization": [
        0.0, 0.01, 1.0, 0.02, 2.0, 0.05, 5.0, 0.1, 10.0, 0.2, 20.0, 0.5,
        50.0, 1.0, 100.0, 2.0, 200.0, 5.0, 500.0,
    ],
    "bandwidth": _SCALAR_DEFAULTS["bandwidth"],
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


def default_init_cops_params(
    instruments: list[str] | tuple[str, ...],
    *,
    depth_is_on: str | None = None,
    number_of_fields_before_date: int = 3,
    windspeed_ms: float = _SCALAR_DEFAULTS["windspeed_ms"],
) -> dict[str, object]:
    """Starting-point ``init.cops.dat`` parameters for a brand-new instrument system.

    Returns a dict shaped exactly like :func:`read_init_cops`'s own output, pre-filled with
    typical values seen across every real deployment cached in this project (see
    ``_INIT_COPS_DAT_INSTRUMENT_DEFAULTS``/``_INIT_COPS_DAT_CONSTANTS``) -- so it round-trips
    through :func:`write_init_cops` unchanged, and a caller only needs to override the handful of
    fields that genuinely vary per project/instrument (``instruments``, ``depth_is_on``,
    ``number_of_fields_before_date``, ``windspeed_ms``, or any per-instrument value in the
    returned dict).

    ``delta.capteur.optics``/``radius.instrument.optics`` are physical sensor properties, not
    universal constants -- a caller presenting these to a researcher (e.g. the cast-cleaning UI's
    generator form) should prompt them to confirm against the instrument's own calibration sheet
    rather than trust the default blindly. ``windspeed_ms`` varies by weather/sea-state on the
    day, not by instrument -- it's usually overridden later per station or even per cast.
    """
    instruments = tuple(instruments)
    unknown = [instr for instr in instruments if instr not in _PER_INSTRUMENT_KEYS_INSTRUMENTS]
    if unknown:
        raise ValueError(
            f"Unknown instrument(s) {unknown}; expected a subset of {_PER_INSTRUMENT_KEYS_INSTRUMENTS}"
        )
    if not instruments:
        raise ValueError("instruments must be non-empty")

    params: dict[str, object] = dict(_INIT_COPS_DAT_CONSTANTS)
    params["instruments.optics"] = list(instruments)
    params["depth.is.on"] = depth_is_on or ("LuZ" if "LuZ" in instruments else instruments[0])
    params["number.of.fields.before.date"] = float(number_of_fields_before_date)
    params["windspeed_ms"] = windspeed_ms
    for name, defaults_by_instrument in _INIT_COPS_DAT_INSTRUMENT_DEFAULTS.items():
        params[name] = {instr: defaults_by_instrument[instr] for instr in instruments}
    return params


# One short, researcher-facing explanation per init.cops.dat parameter -- for a UI form (e.g. the
# cast-cleaning app's generator) to show next to each field.
INIT_COPS_DAT_HELP: dict[str, str] = {
    "instruments.optics": "Optical sensors present on this system (Ed0 = surface irradiance, "
    "always required as a reference).",
    "depth.is.on": "Depth/pressure sensor used as the reference -- normally LuZ.",
    "number.of.fields.before.date": "Number of '_'-separated segments before the date in cast file "
    "names (e.g. 3 for 'SITE_CAST_001_190818_195137_URC.tsv'). pycops usually detects this "
    "automatically from the file name -- this value is only used as a cross-check hint.",
    "windspeed_ms": "Wind speed (m/s) at deployment time -- varies with weather, not with the "
    "instrument; adjust per station or per cast if known.",
    "tiltmax.optics": "Maximum tilt (degrees) beyond which a scan is rejected.",
    "depth.interval.for.smoothing.optics": "Depth interval (m) used for profile smoothing.",
    "sub.surface.removed.layer.optics": "Thickness (m) of the sub-surface layer excluded from "
    "fitting (turbulence/wake near the boat).",
    "delta.capteur.optics": "Depth offset (m) between this sensor and the reference sensor -- a "
    "physical property of the instrument, verify against its calibration sheet.",
    "radius.instrument.optics": "Radius of the sensor housing (m) -- a physical property of the "
    "instrument.",
    "linear.fit.Rsquared.threshold.optics": "Minimum R² to accept the linear surface fit (NA for "
    "Ed0 -- doesn't apply to the surface sensor).",
    "linear.fit.max.delta.depth.optics": "Maximum depth extent (m) of the linear surface fit window "
    "(NA for Ed0).",
    "bandwidth": "Bandwidth (nm) used when computing extraterrestrial irradiance.",
}

# Verbatim header comment block from a real init.cops.dat (local_data/AlgaeWISE/
# 20220705_StationPME6/COPS/init.cops.dat) -- so a freshly generated file looks like one the R
# package's own workflow would produce.
_INIT_COPS_DAT_HEADER = [
    "#" * 127,
    "# lines beginning with # are comments; blank lines are skipped",
    '# EACH LINE HAS THREE FIELDS DELIMITED BY ";"' + " " * 81 + "#",
    "# DO NOT MODIFY THE FIRST TWO FIELDS" + " " * 90 + "#",
    '# IF THE THIRD FIELD CONTAINS SEVERAL VALUES, THESE VALUES ARE SEPARARTED BY ","' + " " * 46 + "#",
    '# IF THE TYPE OF THE THIRD FIELD (GIVEN BY THE SECOND FIELD) IS "character", THE BLANK '
    'CHARACTERS (" ") ARE TAKEN INTO ACCOUNT#',
    "#" * 127,
]


def _format_init_value(name: str, kind: str, value: object, instruments: tuple[str, ...]) -> str:
    if name in _PER_INSTRUMENT_KEYS:
        values = [value[instr] for instr in instruments]
    elif isinstance(value, list):
        values = value
    else:
        values = [value]

    if kind == "logical":
        return ",".join("TRUE" if v else "FALSE" for v in values)
    if kind == "character":
        return ",".join(str(v) for v in values)
    return ",".join("NA" if (isinstance(v, float) and np.isnan(v)) else f"{v:.10g}" for v in values)


# (name, kind, section-comment-or-None) in the same order/grouping as a real init.cops.dat.
_INIT_COPS_DAT_LAYOUT = [
    ("verbose", "logical", None),
    ("indice.water", "numeric", "constants"),
    ("rau.Fresnel", "numeric", None),
    ("win.width", "numeric", "width and height of graphics windows"),
    ("win.height", "numeric", None),
    ("instruments.optics", "character", "optics description"),
    ("tiltmax.optics", "numeric", None),
    ("depth.interval.for.smoothing.optics", "numeric", None),
    ("sub.surface.removed.layer.optics", "numeric", None),
    ("delta.capteur.optics", "numeric", None),
    ("radius.instrument.optics", "numeric", None),
    ("linear.fit.Rsquared.threshold.optics", "numeric", None),
    ("linear.fit.max.delta.depth.optics", "numeric", None),
    ("format.date", "character", "look INTO a data file"),
    ("instruments.others", "character", None),
    ("depth.is.on", "character", None),
    ("number.of.fields.before.date", "numeric", "look at the NAME of a data file"),
    ("time.window", "numeric", "time window : 2 values (unit = second) first = beginning, second = end"),
    ("depth.discretization", "numeric", "for smoothing purpose"),
    ("bandwidth", "numeric", "bandwidth : the size of the window (nanometers) around each wavelength"),
    ("windspeed_ms", "numeric", "Environmental conditions"),
]


def format_init_cops_dat(params: dict[str, object]) -> str:
    """Render an ``init.cops.dat`` params dict (:func:`read_init_cops`'s own shape, e.g. from
    :func:`default_init_cops_params`) back into the file's text format."""
    instruments = tuple(params["instruments.optics"])
    lines = list(_INIT_COPS_DAT_HEADER)
    lines.append("")
    for name, kind, section in _INIT_COPS_DAT_LAYOUT:
        if name not in params:
            continue
        if section:
            lines.append("")
            lines.append(f"# {section} " + "#" * max(0, 50 - len(section)))
        lines.append(f"{name};{kind};{_format_init_value(name, kind, params[name], instruments)}")
    lines.append("")
    return "\r\n".join(lines)


def write_init_cops(path: str | Path, params: dict[str, object], *, overwrite: bool = False) -> None:
    """Write a brand-new ``init.cops.dat`` from a params dict (see :func:`default_init_cops_params`).

    Refuses to clobber an existing file unless ``overwrite=True`` -- unlike
    :func:`update_cast_info`'s surgical per-field edits, this always writes the whole file, so
    silently overwriting one that already has a researcher's own tuned values would be a real
    data-loss risk.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    path.write_text(format_init_cops_dat(params), newline="")


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
    linear_r2_threshold: list[float] | None = None
    linear_max_delta_depth: list[float] | None = None


def parse_override_field(field: str) -> list[float] | None:
    """Parse an ``info.cops.dat`` per-instrument override field: ``"x"``/blank means no override.

    Individual comma-separated values can themselves be ``"NA"`` -- e.g. a linear-fit threshold
    that doesn't apply to the surface (Ed0) instrument, the same sentinel :func:`read_init_cops`
    handles for `init.cops.dat`'s own per-instrument vectors -- confirmed on a real file
    (`local_data/AlgaeWISE/.../info.cops.dat`'s `PME_CAST_019` row: `"NA,0.5,0.5,0.6"`).
    """
    field = field.strip()
    if field in ("", "x"):
        return None
    return [_to_float(v) for v in field.split(",")]


def parse_optional_float(field: str) -> float | None:
    """Parse an ``info.cops.dat`` scalar field: ``"NA"`` (or blank -- a short row padded out by
    :func:`read_info_cops`) means unset."""
    field = field.strip()
    return None if field == "" or field.upper() == "NA" else float(field)


def read_info_cops(path: str | Path) -> list[CastInfo]:
    """Parse an ``info.cops.dat`` file into one :class:`CastInfo` per cast.

    Field layout (1-indexed, matching the file's own documented numbering):
    1 file, 2 longitude, 3 latitude, 4 chl, 5 time.window, 6 sub.surface.removed.layer,
    7 tiltmax, 8 depth.interval.for.smoothing, 9 linear.fit.Rsquared.threshold,
    10 linear.fit.max.delta.depth, 11-14 (optional) dark-measurement file names.
    """
    path = Path(path)
    entries: list[CastInfo] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [f.strip() for f in line.split(";")]
            fields += [""] * (14 - len(fields))

            time_window = parse_override_field(fields[4])

            entries.append(
                CastInfo(
                    file=fields[0],
                    longitude=parse_optional_float(fields[1]),
                    latitude=parse_optional_float(fields[2]),
                    chl_flag=parse_optional_float(fields[3]),
                    time_window=tuple(time_window) if time_window else None,
                    sub_surface_removed_layer=parse_override_field(fields[5]),
                    tiltmax=parse_override_field(fields[6]),
                    depth_interval_for_smoothing=parse_override_field(fields[7]),
                    linear_r2_threshold=parse_override_field(fields[8]),
                    linear_max_delta_depth=parse_override_field(fields[9]),
                    dark_files=[f for f in fields[10:14] if f],
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


# 0-indexed field positions, matching read_info_cops's 1-indexed doc comment above.
_FIELD_INDEX = {
    "longitude": 1,
    "latitude": 2,
    "chl_flag": 3,
    "time_window": 4,
    "sub_surface_removed_layer": 5,
    "tiltmax": 6,
    "depth_interval_for_smoothing": 7,
    "linear_r2_threshold": 8,
    "linear_max_delta_depth": 9,
}
_NA_FIELDS = {"longitude", "latitude", "chl_flag"}  # unset -> "NA"; every other field -> "x"

_UNSET = object()  # update_cast_info's "leave this field alone" sentinel (None means clear it)


def _format_cast_info_field(name: str, value: object) -> str:
    # ".10g" (not the default 6-sig-fig "g"): a plain "g" truncates a real coordinate like
    # -68.11626 to -68.1163, silently losing precision that matters for GPS positions.
    if value is None:
        return "NA" if name in _NA_FIELDS else "x"
    if name == "time_window":
        start, end = value
        return f"{start:.10g},{end:.10g}"
    if name in _NA_FIELDS:
        return f"{value:.10g}"
    # Per-value "NA" (e.g. a linear-fit threshold that doesn't apply to the surface Ed0
    # instrument, see parse_override_field) round-trips as the file's own sentinel, not
    # Python's "nan" repr.
    return ",".join("NA" if np.isnan(v) else f"{v:.10g}" for v in value)


def _extend_row(fields: list[str], max_index: int) -> list[str]:
    fields = list(fields)
    index_to_name = {i: n for n, i in _FIELD_INDEX.items()}
    while len(fields) <= max_index:
        name_at_idx = index_to_name.get(len(fields))
        fields.append("NA" if name_at_idx in _NA_FIELDS else "x")
    return fields


def _dominant_terminator(lines: list[str]) -> str:
    for line in reversed(lines):
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
    return "\r\n"


def update_cast_info(
    path: str | Path,
    file: str,
    *,
    longitude: object = _UNSET,
    latitude: object = _UNSET,
    chl_flag: object = _UNSET,
    time_window: object = _UNSET,
    sub_surface_removed_layer: object = _UNSET,
    tiltmax: object = _UNSET,
    depth_interval_for_smoothing: object = _UNSET,
    linear_r2_threshold: object = _UNSET,
    linear_max_delta_depth: object = _UNSET,
) -> None:
    """Write any subset of an ``info.cops.dat`` row's fields (2-10), leaving the rest untouched.

    A surgical line-level edit rather than a full read-into-:class:`CastInfo`-then-rewrite-
    everything round trip, so it never disturbs a row/field/comment it doesn't own -- real files
    carry a large hand-written comment header. Real ``info.cops.dat`` files are CRLF-terminated;
    this preserves whatever line ending each existing line already has (untouched lines round-trip
    byte-for-byte) rather than normalizing the whole file.

    Every keyword defaults to a private "leave alone" sentinel -- only fields explicitly passed
    are written. Pass ``None`` to explicitly clear a field back to its ``"NA"``/``"x"`` sentinel
    (``"NA"`` for ``longitude``/``latitude``/``chl_flag``, ``"x"`` -- "use init.cops.dat's
    deployment-wide default" -- for the rest). ``time_window`` is ``(start, end)`` in elapsed
    seconds; the per-instrument override fields are ``list[float]`` (one value per
    ``instruments.optics`` entry).

    If ``file`` already has a row, it's extended (with ``"NA"``/``"x"`` placeholders as
    appropriate) only as far as the highest field index among the given updates, then those
    fields are replaced in place. If the file exists but has no row for ``file``, one is appended
    (matching a real minimal row's shape). If ``path`` doesn't exist at all, it's created with the
    same header comment block a real ``cops.go()``-generated file has, plus that one row.
    """
    updates = {
        name: value
        for name, value in (
            ("longitude", longitude),
            ("latitude", latitude),
            ("chl_flag", chl_flag),
            ("time_window", time_window),
            ("sub_surface_removed_layer", sub_surface_removed_layer),
            ("tiltmax", tiltmax),
            ("depth_interval_for_smoothing", depth_interval_for_smoothing),
            ("linear_r2_threshold", linear_r2_threshold),
            ("linear_max_delta_depth", linear_max_delta_depth),
        )
        if value is not _UNSET
    }
    if not updates:
        return

    path = Path(path)
    max_index = max(_FIELD_INDEX[name] for name in updates)

    def _apply(fields: list[str]) -> list[str]:
        fields = _extend_row(fields, max_index)
        for name, value in updates.items():
            fields[_FIELD_INDEX[name]] = _format_cast_info_field(name, value)
        return fields

    if not path.exists():
        row = _apply([file])
        content = "\r\n".join([*_INFO_COPS_DAT_HEADER, ";".join(row)]) + "\r\n"
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
        lines[i] = ";".join(_apply(fields)) + terminator
        found = True
        break

    if not found:
        terminator = _dominant_terminator(lines)
        if lines and not lines[-1].endswith(("\n", "\r\n", "\r")):
            lines[-1] += terminator
        lines.append(";".join(_apply([file])) + terminator)

    with path.open("w", newline="") as f:
        f.write("".join(lines))


def update_time_window(path: str | Path, file: str, time_window: tuple[float, float]) -> None:
    """Write a per-cast ``time.window`` override -- a thin, single-field wrapper over
    :func:`update_cast_info`, kept as its own function since it's this app's most common edit."""
    update_cast_info(path, file, time_window=time_window)


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
