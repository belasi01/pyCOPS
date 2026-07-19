"""Resolve a cast's position (longitude/latitude) when ``info.cops.dat`` lacks it.

Ports the GPS-file half of ``derived.data.R``'s position-resolution chain:
when a cast's ``info.cops.dat`` longitude/latitude is ``NA``, look for a
``GPS_*.tsv``/``.csv`` file in the same deployment folder and take the median
position of GPS fixes recorded during the cast's own time window. The
``BioGPS.Position`` raw-column fallback (an embedded ancillary GPS reading in
some older raw files, used when there's no separate GPS file) is not ported --
no real sample was available to validate the decoding against.

Position and time can also be measured wrong or not at all -- the COPS's own
GPS has failed in the field before, a real recurring situation. For that case
:func:`~pycops.processing.process_cast.process_cast` and
:func:`~pycops.processing.deployment.process_deployment` accept explicit
:class:`PositionOverride` values (longitude/latitude/UTC time supplied
directly by the researcher) that take priority over both ``info.cops.dat``
and any GPS file -- a pycops-only escape hatch the R package doesn't have (it
only tells the user, via a printed message, to hand-edit ``info.cops.dat``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PositionOverride:
    """Manual longitude/latitude/UTC-time override for one cast.

    Use when a cast's own position/time (``info.cops.dat``, any GPS file, and
    the cast's recorded timestamps) is unavailable or known to be wrong --
    e.g. the COPS's GPS failed in the field. Any field left ``None`` falls
    back to the normal resolution chain (``info.cops.dat`` / GPS file / the
    cast's own recorded time).
    """

    longitude: float | None = None
    latitude: float | None = None
    utc_time: pd.Timestamp | None = None


def read_gps_file(path: str | Path) -> pd.DataFrame:
    """Parse a uProfile ``GPS_*.tsv``/``.csv`` file into a DataFrame.

    Returns columns ``time`` (parsed from ``DateTimeUTC``), ``longitude``,
    ``latitude``. Port of the "uprofile 1.9.10 and after" branch of
    ``derived.data.R``'s GPS-file reader (tab-separated ``GPS_*.tsv``/``.txt``
    files, comma-separated otherwise); the pre-2016 legacy column layout it
    also handles is not ported (no real sample was available to validate
    against). ``DateTimeUTC``'s sub-second precision is inconsistent row to
    row in real files (some rows carry milliseconds, some don't), so parsing
    uses pandas' per-row format inference rather than one fixed format string.
    """
    path = Path(path)
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    table = pd.read_csv(path, sep=sep, quotechar='"')
    table.columns = [c.strip("[]") for c in table.columns]
    return pd.DataFrame(
        {
            "time": pd.to_datetime(table["DateTimeUTC"], format="mixed"),
            "longitude": table["Longitude"].astype(float),
            "latitude": table["Latitude"].astype(float),
        }
    )


def find_gps_file(directory: str | Path) -> Path | None:
    """First ``GPS_*`` file in a deployment folder, or ``None`` if there isn't one.

    Real deployments observed so far carry at most one GPS file per folder;
    ``derived.data.R``'s handling of multiple candidate GPS files (matched by
    a date token in the cast's own file name) isn't ported.
    """
    matches = sorted(Path(directory).glob("GPS_*"))
    return matches[0] if matches else None


def position_from_gps(gps_table: pd.DataFrame, cast_times: np.ndarray) -> tuple[float, float] | None:
    """Median GPS longitude/latitude recorded during a cast's own time window.

    ``cast_times`` is a cast's own ``time`` coordinate; its min/max bound the
    search window, matching ``derived.data.R``'s ``dates`` range check.
    Returns ``None`` if no GPS fix falls in that window (the cast's clock and
    the GPS file's clock don't overlap -- ``derived.data.R`` just warns and
    leaves the position unresolved in this case, it doesn't raise).
    """
    cast_times = np.asarray(cast_times)
    start, end = cast_times.min(), cast_times.max()
    valid = (gps_table["time"] >= start) & (gps_table["time"] <= end)
    if not valid.any():
        return None
    return float(gps_table.loc[valid, "longitude"].median()), float(gps_table.loc[valid, "latitude"].median())
