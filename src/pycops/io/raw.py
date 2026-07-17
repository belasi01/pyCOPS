"""Read raw C-OPS profiler cast files into xarray Datasets.

Ports the parsing logic of the ``Cops`` R package
(``read.COPS.R`` / ``read.data.R``, Bernard Gentili / Simon Belanger)
to Python. A cast file is a CSV or TSV export from Biospherical's
uProfile software: one row per scan, with spectral channels named
``<instrument><wavelength>`` (e.g. ``Ed0443``) and ancillary channels
named ``<instrument><name>`` (e.g. ``LuZDepth``, ``Ed0Roll``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_INSTRUMENTS = ("Ed0", "EdZ", "LuZ", "EuZ")


@dataclass(frozen=True)
class CastFileInfo:
    """Metadata recovered from a cast file name, without reading its content."""

    path: Path
    cast_number: str
    date: datetime
    is_urc: bool
    gps_file: Path | None


def parse_cast_filename(path: str | Path, number_of_fields_before_date: int) -> CastFileInfo:
    """Extract the cast date/time, cast number, and matching GPS file from a file name.

    ``number_of_fields_before_date`` comes from the ``init.cops.dat`` file of the
    deployment and tells how many ``_``-separated tokens precede the ``YYMMDD``
    date token in the file name (e.g. 3 for ``WISE_CAST_001_190817_220856_URC.csv``).
    """
    path = Path(path)
    parts = path.name.split("_")
    is_urc = "URC." in path.name

    n = number_of_fields_before_date
    date_token = parts[n]
    time_token = parts[n + 1]
    # File names encode time as HHMM or HHMMSS; only HH and MM are used, matching
    # the R package (a per-scan timestamp is recovered later from the data itself).
    date = datetime.strptime(f"{date_token[:6]}{time_token[:4]}", "%y%m%d%H%M")

    if is_urc:
        cast_number = parts[n - 1]
        gps_stem = f"GPS_{date_token}"
    else:
        cast_number = parts[n + 2]
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
    number_of_fields_before_date: int = 3,
) -> xr.Dataset:
    """Read one raw C-OPS cast file (CSV or TSV) into an ``xarray.Dataset``.

    Spectral channels (e.g. ``Ed0``, ``EdZ``, ``LuZ``, ``EuZ``) become data
    variables with dims ``(time, wavelength)``; a shared ``wavelength`` dim is
    used when all instruments share the same bands, otherwise each instrument
    gets its own ``wavelength_<instr>`` dim. Ancillary channels (roll, pitch,
    depth, temperature) become ``<instrument>_<name>`` variables on ``time``,
    and unmatched columns (GPS/BioShade fields, raw timestamps) are kept as-is.
    """
    path = Path(path)
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
