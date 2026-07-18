"""Read raw C-OPS profiler cast files into xarray Datasets.

Ports the parsing logic of the ``Cops`` R package
(``read.COPS.R`` / ``read.data.R``, Bernard Gentili / Simon Belanger)
to Python. A cast file is a CSV or TSV export from Biospherical's
uProfile software: one row per scan, with spectral channels named
``<instrument><wavelength>`` (e.g. ``Ed0443``) and ancillary channels
named ``<instrument><name>`` (e.g. ``LuZDepth``, ``Ed0Roll``).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_INSTRUMENTS = ("Ed0", "EdZ", "LuZ", "EuZ")

_DATE_RE = re.compile(r"^\d{6}$")
_TIME_RE = re.compile(r"^(?:\d{4}|\d{6})$")


@dataclass(frozen=True)
class CastFileInfo:
    """Metadata recovered from a cast file name, without reading its content."""

    path: Path
    cast_number: str
    date: datetime
    is_urc: bool
    gps_file: Path | None


def _is_plausible_date_token(token: str) -> bool:
    """Does ``token`` look like a ``YYMMDD`` date (loose range checks, not calendar-exact)?"""
    if not _DATE_RE.match(token):
        return False
    month, day = int(token[2:4]), int(token[4:6])
    return 1 <= month <= 12 and 1 <= day <= 31


def _is_plausible_time_token(token: str) -> bool:
    """Does ``token`` look like an ``HHMM`` or ``HHMMSS`` time?"""
    if not _TIME_RE.match(token):
        return False
    hour, minute = int(token[:2]), int(token[2:4])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return False
    if len(token) == 6:
        return 0 <= int(token[4:6]) <= 59
    return True


def _find_date_time_candidates(parts: list[str]) -> list[int]:
    """Indices ``i`` where ``parts[i]``/``parts[i + 1]`` look like a date/time pair."""
    return [
        i
        for i in range(len(parts) - 1)
        if _is_plausible_date_token(parts[i]) and _is_plausible_time_token(parts[i + 1])
    ]


def _resolve_date_time_position(parts: list[str], filename: str, number_of_fields_before_date: int | None) -> int:
    """Find which token in ``parts`` is the date, auto-detecting from the filename's shape.

    ``number_of_fields_before_date`` (from ``init.cops.dat``, already adjusted for ``_SB_``)
    is used only as an optional cross-check/tie-breaker -- real deployments sometimes have a
    wrong value for the files sitting next to it, so a plausible date/time token pair actually
    found in the filename always wins over a configured position that doesn't match one.
    """
    candidates = _find_date_time_candidates(parts)
    hint = None
    if number_of_fields_before_date is not None and 0 <= number_of_fields_before_date < len(parts) - 1:
        hint = number_of_fields_before_date

    if not candidates:
        raise ValueError(
            f"could not find a YYMMDD/HHMM(SS) date-time token pair in {filename!r} "
            f"(tokens={parts!r}); expected a 6-digit date token immediately followed by "
            f"a 4- or 6-digit time token"
        )

    if len(candidates) == 1:
        detected = candidates[0]
        if hint is not None and hint != detected:
            warnings.warn(
                f"{filename}: number.of.fields.before.date points at token {hint} "
                f"({parts[hint]!r}), but token {detected} ({parts[detected]!r}, followed by "
                f"{parts[detected + 1]!r}) is the one that actually looks like a date/time -- "
                f"using the detected position. This deployment's init.cops.dat "
                f"number.of.fields.before.date looks wrong for this file.",
                stacklevel=3,
            )
        return detected

    if hint in candidates:
        return hint

    raise ValueError(
        f"ambiguous date-time position in {filename!r}: multiple token pairs look plausible "
        f"({[(i, parts[i], parts[i + 1]) for i in candidates]!r}); pass a "
        f"number_of_fields_before_date matching one of these positions to disambiguate"
    )


def _extract_legacy_cast_number(parts: list[str], time_idx: int) -> str:
    """Best-effort cast number for non-URC (legacy) file names, from the tail after the time token.

    The last purely-numeric tail token is usually the cast number (e.g. ``..._data_001.tsv``);
    falls back to the first tail token if none is numeric, rather than raising -- this field is
    informational only.
    """
    tail = list(parts[time_idx + 1 :])
    if not tail:
        return ""
    tail[-1] = Path(tail[-1]).stem  # strip the extension from the final token only
    numeric_tail = [t for t in tail if t.isdigit()]
    return numeric_tail[-1] if numeric_tail else tail[0]


def parse_cast_filename(path: str | Path, number_of_fields_before_date: int | None = None) -> CastFileInfo:
    """Extract the cast date/time, cast number, and matching GPS file from a file name.

    The date/time token position is auto-detected from the filename's shape (a plausible
    ``YYMMDD`` token immediately followed by a plausible ``HHMM``/``HHMMSS`` token), since real
    deployments' filename shapes vary a lot (from 1 to 6 ``_``-separated tokens before the date
    across projects) and their ``init.cops.dat`` ``number.of.fields.before.date`` is sometimes
    simply wrong for the files next to it. ``number_of_fields_before_date`` is an optional
    cross-check/tie-breaker, not authoritative -- see :func:`_resolve_date_time_position`.
    BioShade casts are named with ``_SB_`` instead of ``_CAST_NNN_`` (e.g.
    ``hudsonbay_SB_180605_192518_URC.csv``) -- one fewer token before the date -- and are
    detected and adjusted for automatically.
    """
    path = Path(path)
    parts = path.name.split("_")
    is_urc = "URC." in path.name

    hint = number_of_fields_before_date
    if hint is not None and "_SB_" in path.name:
        hint -= 1
    n = _resolve_date_time_position(parts, path.name, hint)

    date_token = parts[n]
    time_token = parts[n + 1]
    # File names encode time as HHMM or HHMMSS; only HH and MM are used, matching
    # the R package (a per-scan timestamp is recovered later from the data itself).
    date = datetime.strptime(f"{date_token[:6]}{time_token[:4]}", "%y%m%d%H%M")

    if is_urc:
        cast_number = parts[n - 1] if n > 0 else ""
        gps_stem = f"GPS_{date_token}"
    else:
        cast_number = _extract_legacy_cast_number(parts, n + 1)
        gps_stem = "_".join(parts[: n + 2]) + "_gps"

    matches = sorted(path.parent.glob(f"{gps_stem}*"))
    gps_file = matches[0] if matches else None

    return CastFileInfo(path=path, cast_number=cast_number, date=date, is_urc=is_urc, gps_file=gps_file)


def _detect_header_lines(path: Path) -> int:
    """Count lines to skip when the file starts with a ``Start of Header`` block."""
    with path.open(encoding="latin-1") as f:
        first = f.readline().strip()
        if first != "Start of Header":
            return 0
        n = 1
        for line in f:
            n += 1
            if line.strip() == "End of Header":
                break
        return n


def _clean_column_names(columns: list[str]) -> list[str]:
    """Strip unit suffixes and legacy bracket notation from raw column headers.

    uProfile exports column names as e.g. ``Ed0443 (uW/(cm^2 nm))``; only the
    token before the first space is kept. Files from ~2011 instead bracket the
    wavelength, e.g. ``Ed0[443]`` -- matching the R package, this legacy form is
    detected by inspecting only the *first* column name.
    """
    if "]" in columns[0]:
        columns = [c.replace("[", "").replace("]", "") for c in columns]
    return [c.split(" ")[0] for c in columns]


def _split_instruments(
    df: pd.DataFrame, instruments: tuple[str, ...]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, pd.DataFrame], pd.DataFrame]:
    """Split columns into per-instrument spectral matrices, ancillary frames, and leftovers."""
    spectral: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ancillary: dict[str, pd.DataFrame] = {}
    consumed: list[str] = []

    for instr in instruments:
        instr_cols = [c for c in df.columns if c.startswith(instr)]
        if not instr_cols:
            # Instrument not present on this deployment (e.g. no EuZ sensor).
            continue
        consumed.extend(instr_cols)
        suffixes = [c[len(instr) :].lstrip(":") for c in instr_cols]

        wave_cols = [c for c, s in zip(instr_cols, suffixes) if s.isdigit()]
        wave_suffixes = [s for s in suffixes if s.isdigit()]
        anc_cols = [c for c, s in zip(instr_cols, suffixes) if not s.isdigit()]
        anc_suffixes = [s for s in suffixes if not s.isdigit()]

        waves = np.asarray(wave_suffixes, dtype=float)
        order = np.argsort(waves)
        values = df[wave_cols].to_numpy(dtype=float)[:, order]
        spectral[instr] = (waves[order], values)

        anc_df = df[anc_cols].copy()
        anc_df.columns = anc_suffixes
        ancillary[instr] = anc_df

    others = df.drop(columns=consumed)
    return spectral, ancillary, others


def _infer_time(others: pd.DataFrame, fallback_date: datetime, n_rows: int) -> np.ndarray:
    """Recover a per-scan timestamp from the ``Others`` columns.

    Tries, in order: ``DateTimeUTC`` (sub-second precision), ``DateTime``,
    then the Excel serial date in ``GeneralExcelTime``. Falls back to the
    single timestamp encoded in the file name if none are usable.
    """
    for col in ("DateTimeUTC", "DateTime"):
        if col in others.columns:
            parsed = pd.to_datetime(others[col], errors="coerce", format="mixed")
            if parsed.notna().all():
                return parsed.to_numpy()

    if "GeneralExcelTime" in others.columns:
        excel_epoch = pd.Timestamp("1899-12-30")
        parsed = excel_epoch + pd.to_timedelta(others["GeneralExcelTime"].astype(float), unit="D")
        if parsed.notna().all():
            return parsed.to_numpy()

    return np.full(n_rows, np.datetime64(fallback_date), dtype="datetime64[ns]")


def _all_waves_equal(spectral: dict[str, tuple[np.ndarray, np.ndarray]]) -> bool:
    arrays = [waves for waves, _ in spectral.values()]
    return all(np.array_equal(arrays[0], a) for a in arrays[1:])


def read_cast(
    path: str | Path,
    instruments: tuple[str, ...] = DEFAULT_INSTRUMENTS,
    number_of_fields_before_date: int | None = None,
) -> xr.Dataset:
    """Read one raw C-OPS cast file (CSV or TSV) into an ``xarray.Dataset``.

    Spectral channels (e.g. ``Ed0``, ``EdZ``, ``LuZ``, ``EuZ``) become data
    variables with dims ``(time, wavelength)``; a shared ``wavelength`` dim is
    used when all instruments share the same bands, otherwise each instrument
    gets its own ``wavelength_<instr>`` dim. Ancillary channels (roll, pitch,
    depth, temperature) become ``<instrument>_<name>`` variables on ``time``,
    and unmatched columns (GPS/BioShade fields, raw timestamps) are kept as-is.
    ``number_of_fields_before_date`` is an optional cross-check hint for the
    file name's date/time position (auto-detected otherwise) -- see
    :func:`parse_cast_filename`. BioShade casts (``_SB_`` in the file name)
    only carry an Ed0 sensor, so
    ``instruments`` is forced to ``("Ed0",)`` for them regardless of what's
    passed in.
    """
    path = Path(path)
    if "_SB_" in path.name:
        instruments = ("Ed0",)
    info = parse_cast_filename(path, number_of_fields_before_date)

    n_header = _detect_header_lines(path)
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=delimiter, skiprows=n_header, encoding="latin-1")
    df.columns = _clean_column_names(list(df.columns))

    spectral, ancillary, others = _split_instruments(df, instruments)
    time = _infer_time(others, info.date, len(df))

    shared_wavelength = _all_waves_equal(spectral)
    coords: dict[str, object] = {"time": time}
    data_vars: dict[str, object] = {}

    for instr, (waves, values) in spectral.items():
        dim = "wavelength" if shared_wavelength else f"wavelength_{instr}"
        coords.setdefault(dim, waves)
        data_vars[instr] = (("time", dim), values)

    for instr, anc_df in ancillary.items():
        for col in anc_df.columns:
            data_vars[f"{instr}_{col}"] = ("time", anc_df[col].to_numpy())

    for col in others.columns:
        data_vars[col] = ("time", others[col].to_numpy())

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "source_file": str(info.path),
            "cast_number": info.cast_number,
            "cast_date": info.date.isoformat(),
            "is_urc_format": info.is_urc,
            "gps_file": str(info.gps_file) if info.gps_file else "",
            "instruments": list(spectral.keys()),
        },
    )
