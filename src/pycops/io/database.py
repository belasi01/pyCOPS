"""Persist a :class:`~pycops.processing.database.MissionDatabase` to NetCDF and CSV.

NetCDF is pycops's own established self-describing-format replacement for R's ``.RData``
(see ``CLAUDE.md``'s "Output format" note), so :func:`write_mission_database_netcdf` is the
pycops-native parity for ``generate.cops.DB.R``'s ``COPS.DB.RData``; :func:`write_mission_database_csv`
is the parity for its comma-separated ``.dat`` file, for direct spreadsheet/legacy-workflow use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pycops.processing.database import MissionDatabase, StationAggregate

# (attribute name on StationAggregate, output-name prefix) for every per-wavelength MeanSd metric.
_WAVELENGTH_METRICS = (
    ("rrs", "Rrs"),
    ("nlw", "nLw"),
    ("rb", "Rb"),
    ("kd_1pct", "Kd1pct"),
    ("kd_10pct", "Kd10pct"),
    ("kd_pd", "Kdpd"),
    ("ed0_0p", "Ed0.0p"),
)

# (attribute name on StationAggregate, output-name prefix) for every scalar (non-wavelength) PAR/
# Kd(PAR) metric -- broadband, so unlike _WAVELENGTH_METRICS these get one mean/sd column each,
# not one per wavelength.
_SCALAR_METRICS = (
    ("par_0", "PAR0"),
    ("kd_par_1pct", "KdPAR1pct"),
    ("kd_par_10pct", "KdPAR10pct"),
    ("kd_par_pd", "KdPARpd"),
)


def _station_field(stations: list[StationAggregate], attr: str, missing: object) -> np.ndarray:
    return np.array([getattr(s, attr) if getattr(s, attr) is not None else missing for s in stations])


def write_mission_database_netcdf(db: MissionDatabase, path: str | Path) -> None:
    """One NetCDF file, dims ``(station, wavelength)``, for the whole mission database."""
    n_stations = len(db.stations)
    coords = {
        "station": np.arange(n_stations),
        "wavelength": db.waves,
        "station_id": ("station", [s.station_id for s in db.stations]),
    }
    data_vars: dict[str, tuple] = {}

    for attr, prefix in _WAVELENGTH_METRICS:
        mean_stack = np.array([getattr(s, attr).mean for s in db.stations]) if n_stations else np.empty((0, len(db.waves)))
        sd_stack = np.array([getattr(s, attr).sd for s in db.stations]) if n_stations else np.empty((0, len(db.waves)))
        data_vars[f"{prefix}_mean"] = (("station", "wavelength"), mean_stack)
        data_vars[f"{prefix}_sd"] = (("station", "wavelength"), sd_stack)

    data_vars["Ed0_diffuse_fraction"] = (
        ("station", "wavelength"),
        np.array([s.ed0_diffuse_fraction for s in db.stations])
        if n_stations
        else np.empty((0, len(db.waves))),
    )
    data_vars["n_casts"] = ("station", _station_field(db.stations, "n_casts", 0))
    data_vars["date"] = (
        "station",
        np.array([s.date_mean if s.date_mean is not None else np.datetime64("NaT") for s in db.stations]),
    )
    data_vars["sun_zenith_deg"] = ("station", _station_field(db.stations, "sun_zenith_mean", np.nan))
    data_vars["longitude"] = ("station", _station_field(db.stations, "longitude_mean", np.nan))
    data_vars["latitude"] = ("station", _station_field(db.stations, "latitude_mean", np.nan))
    data_vars["forel_ule"] = ("station", _station_field(db.stations, "forel_ule_mean", np.nan))
    data_vars["shadow_correction_method"] = (
        "station",
        _station_field(db.stations, "shadow_correction_method", ""),
    )
    data_vars["bottom_depth"] = ("station", _station_field(db.stations, "bottom_depth_mean", np.nan))
    data_vars["directory"] = ("station", [str(s.directory) for s in db.stations])

    for attr, prefix in _SCALAR_METRICS:
        mean_arr = np.array([getattr(s, attr).mean for s in db.stations]) if n_stations else np.empty(0)
        sd_arr = np.array([getattr(s, attr).sd for s in db.stations]) if n_stations else np.empty(0)
        data_vars[f"{prefix}_mean"] = ("station", mean_arr)
        data_vars[f"{prefix}_sd"] = ("station", sd_arr)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={"mission": db.mission})
    ds.to_netcdf(Path(path), engine="netcdf4")


def write_mission_database_csv(db: MissionDatabase, path: str | Path) -> None:
    """One row per station, columns named ``<Metric>_<wave>_mean``/``_sd`` -- matches
    ``generate.cops.DB.R``'s own ``.dat`` column-naming convention (e.g. ``Rrs_443_mean``)."""
    rows = []
    for station in db.stations:
        row: dict[str, object] = {
            "station_id": station.station_id,
            "directory": str(station.directory),
            "n_casts": station.n_casts,
            "date": station.date_mean,
            "sun_zenith_deg": station.sun_zenith_mean,
            "longitude": station.longitude_mean,
            "latitude": station.latitude_mean,
            "forel_ule": station.forel_ule_mean,
            "shadow_correction_method": station.shadow_correction_method,
            "bottom_depth": station.bottom_depth_mean,
        }
        for attr, prefix in _SCALAR_METRICS:
            scalar = getattr(station, attr)
            row[f"{prefix}_mean"] = scalar.mean
            row[f"{prefix}_sd"] = scalar.sd
        for attr, prefix in _WAVELENGTH_METRICS:
            mean_sd = getattr(station, attr)
            for wave, mean_value in zip(db.waves, mean_sd.mean):
                row[f"{prefix}_{wave:g}_mean"] = mean_value
            for wave, sd_value in zip(db.waves, mean_sd.sd):
                row[f"{prefix}_{wave:g}_sd"] = sd_value
        for wave, value in zip(db.waves, station.ed0_diffuse_fraction):
            row[f"Ed0.f.diff_{wave:g}"] = value
        rows.append(row)

    pd.DataFrame(rows).to_csv(Path(path), index=False)
