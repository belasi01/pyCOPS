"""Read the ``init.cops.dat`` and ``info.cops.dat`` configuration files.

These sit alongside raw cast files in every ``COPS*/`` deployment folder.
``init.cops.dat`` holds processing parameters for the whole deployment;
``info.cops.dat`` is a table with one row per cast file giving its
position and any per-cast overrides. Ports ``read.init.R`` from the
``Cops`` R package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

_CASTERS = {
    "numeric": lambda values: [float(v) for v in values],
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

    return params


@dataclass(frozen=True)
class CastInfo:
    """One row of ``info.cops.dat``: a cast file's position and overrides."""

    file: str
    longitude: float
    latitude: float
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

            chl_field = fields[3]
            time_window = _override(fields[4])

            entries.append(
                CastInfo(
                    file=fields[0],
                    longitude=float(fields[1]),
                    latitude=float(fields[2]),
                    chl_flag=None if chl_field.upper() == "NA" else float(chl_field),
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
