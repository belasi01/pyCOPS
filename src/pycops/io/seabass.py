"""SeaBASS-compliant per-station text export, "comme le fait HyperCP" (NASA's own hyperspectral
processor, `github.com/nasa/HyperCP`).

SeaBASS's own data model is inherently per-station (one file's header describes one station's
date/position/etc -- there's no multi-station SeaBASS file), so :func:`write_seabass_station_file`
writes one ``.sb`` text file per aggregated :class:`~pycops.processing.database.StationAggregate`,
not one mission-wide file. The header block/``/fields``/``/units``/``missing``-sentinel convention
is modeled directly on NASA HyperCP's real writer (``Source/SeaBASSWriter.py``/
``SeaBASSHeader.py``, read directly rather than guessed at from general docs): ``/begin_header``,
one ``/{key}={value}`` line per header field, then ``/fields=``/``/units=``, then ``/end_header``,
then one comma-delimited data row.

The ``Kd1pct``/``Kd10pct``/``Kdpd`` fields are pycops/R-specific derived metrics with no SeaBASS
standard-vocabulary name -- flagged with an explanatory ``!`` comment line, matching how real
SeaBASS submissions commonly annotate non-standard fields for the archive's review process.
``data_type``'s default (``"above_water"``, SeaBASS's keyword for station-level AOP products)
should be double-checked against SeaBASS's live ``/data_type`` controlled vocabulary before a
real submission -- it was not independently reachable while building this from documentation
alone.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from pycops.processing.database import StationAggregate

# (StationAggregate metric attr, SeaBASS field-name prefix, units) -- one row per wavelength-keyed
# metric written to the data row. Kd metrics have no SeaBASS standard name (see module docstring).
_WAVELENGTH_FIELDS = (
    ("rrs", "Rrs", "1/sr"),
    ("nlw", "nLw", "uW/cm^2/nm/sr"),
    ("ed0_0p", "Es", "uW/cm^2/nm"),
    ("rb", "Rb", "unitless"),
    ("kd_1pct", "Kd1pct", "1/m"),
    ("kd_10pct", "Kd10pct", "1/m"),
    ("kd_pd", "Kdpd", "1/m"),
)


@dataclass
class SeaBASSHeaderFields:
    """Mission-wide metadata pycops has no other source for -- supplied once (e.g. via a UI
    form) and applied to every station's ``.sb`` file. Station/date/position/data_file_name are
    filled in automatically per station by :func:`write_seabass_station_file`, not here."""

    investigators: str = ""
    affiliations: str = ""
    contact: str = ""
    experiment: str = ""
    cruise: str = ""
    documents: str = "NA"
    calibration_files: str = "NA"
    data_type: str = "above_water"
    water_depth: str = "NA"
    missing: int = -9999
    delimiter: str = "comma"


def _format_value(value: float | None, missing: int) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return str(missing)
    return f"{value:.6g}"


def write_seabass_station_file(
    aggregate: StationAggregate,
    header: SeaBASSHeaderFields,
    waves: np.ndarray,
    path: str | Path,
) -> None:
    """Write one SeaBASS-formatted ``.sb`` file for ``aggregate`` at ``path``.

    ``waves`` is the mission's shared wavelength grid (``MissionDatabase.waves``, after any
    all-NaN bands were already dropped) -- one ``mean``/``sd`` column pair per metric per
    wavelength, ``NaN`` replaced by ``header.missing``.
    """
    path = Path(path)
    date_str = aggregate.date_mean.strftime("%Y%m%d") if aggregate.date_mean is not None else "NA"
    time_str = aggregate.date_mean.strftime("%H:%M:%S") if aggregate.date_mean is not None else "NA"
    lat_str = _format_value(aggregate.latitude_mean, header.missing)
    lon_str = _format_value(aggregate.longitude_mean, header.missing)

    header_lines = ["/begin_header"]
    for f in fields(header):
        header_lines.append(f"/{f.name}={getattr(header, f.name)}")
    header_lines.append(f"/station={aggregate.station_id}")
    header_lines.append(f"/data_file_name={path.name}")
    header_lines.append(f"/start_date={date_str}")
    header_lines.append(f"/end_date={date_str}")
    header_lines.append(f"/start_time={time_str}")
    header_lines.append(f"/end_time={time_str}")
    header_lines.append(f"/north_latitude={lat_str}[DEG]")
    header_lines.append(f"/south_latitude={lat_str}[DEG]")
    header_lines.append(f"/east_longitude={lon_str}[DEG]")
    header_lines.append(f"/west_longitude={lon_str}[DEG]")
    header_lines.append(
        "!Kd1pct/Kd10pct/Kdpd are pycops-derived (mean diffuse attenuation from the surface to "
        "the 1%/10%/penetration-depth light levels), not standard SeaBASS fields."
    )

    field_names = ["date", "time", "lat", "lon", "station"]
    units = ["yyyymmdd", "hh:mm:ss", "degrees", "degrees", "none"]
    row_values = [date_str, time_str, lat_str, lon_str, aggregate.station_id]

    for attr, prefix, unit in _WAVELENGTH_FIELDS:
        mean_sd = getattr(aggregate, attr)
        for wave, mean_value, sd_value in zip(waves, mean_sd.mean, mean_sd.sd):
            wave_label = f"{wave:g}"
            field_names.append(f"{prefix}{wave_label}")
            units.append(unit)
            row_values.append(_format_value(mean_value, header.missing))
            field_names.append(f"{prefix}{wave_label}_sd")
            units.append(unit)
            row_values.append(_format_value(sd_value, header.missing))

    header_lines.append(f"/fields={','.join(field_names)}")
    header_lines.append(f"/units={','.join(units)}")
    header_lines.append("/end_header")

    with path.open("w") as f:
        f.write("\n".join(header_lines) + "\n")
        f.write(",".join(row_values) + "\n")
