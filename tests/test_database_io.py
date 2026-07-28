from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pycops.io.database import write_mission_database_csv, write_mission_database_netcdf
from pycops.processing.database import MeanSd, MissionDatabase, StationAggregate

WAVES = np.array([443.0, 555.0])


def _make_station(station_id, rrs_mean, ed0_mean, n_casts=2):
    n = len(WAVES)
    zeros = MeanSd(mean=np.zeros(n), sd=np.zeros(n))
    return StationAggregate(
        station_id=station_id,
        directory=Path(f"/data/{station_id}"),
        n_casts=n_casts,
        date_mean=pd.Timestamp("2019-08-17T12:00:00"),
        sun_zenith_mean=45.0,
        longitude_mean=-68.1,
        latitude_mean=49.1,
        forel_ule_mean=6.0,
        shadow_correction_method="abs.Kd.method",
        bottom_depth_mean=None,
        ed0_diffuse_fraction=np.array([0.4, 0.45]),
        rrs=MeanSd(mean=np.asarray(rrs_mean, dtype=float), sd=np.array([0.01, 0.02])),
        nlw=zeros,
        rb=MeanSd(mean=np.full(n, np.nan), sd=np.full(n, np.nan)),
        kd_1pct=zeros,
        kd_10pct=zeros,
        kd_pd=zeros,
        ed0_0p=MeanSd(mean=np.asarray(ed0_mean, dtype=float), sd=np.array([1.0, 2.0])),
    )


def _make_db():
    return MissionDatabase(
        mission="TestMission",
        waves=WAVES,
        stations=[
            _make_station("A", rrs_mean=[1.0, 10.0], ed0_mean=[100.0, 200.0]),
            _make_station("B", rrs_mean=[2.0, 20.0], ed0_mean=[110.0, 210.0]),
        ],
    )


def test_write_mission_database_netcdf_round_trips(tmp_path):
    db = _make_db()
    path = tmp_path / "mission.nc"

    write_mission_database_netcdf(db, path)
    reloaded = xr.open_dataset(path)
    try:
        assert reloaded.attrs["mission"] == "TestMission"
        assert list(reloaded["station_id"].values) == ["A", "B"]
        np.testing.assert_allclose(reloaded["Rrs_mean"].values, [[1.0, 10.0], [2.0, 20.0]])
        np.testing.assert_allclose(reloaded["Ed0.0p_mean"].values, [[100.0, 200.0], [110.0, 210.0]])
        np.testing.assert_allclose(reloaded["wavelength"].values, WAVES)
        assert bool(np.all(np.isnan(reloaded["Rb_mean"].values)))
    finally:
        reloaded.close()


def test_write_mission_database_csv_column_naming(tmp_path):
    db = _make_db()
    path = tmp_path / "mission.csv"

    write_mission_database_csv(db, path)
    df = pd.read_csv(path)

    assert list(df["station_id"]) == ["A", "B"]
    assert "Rrs_443_mean" in df.columns
    assert "Rrs_443_sd" in df.columns
    assert "Ed0.0p_555_mean" in df.columns
    np.testing.assert_allclose(df["Rrs_443_mean"], [1.0, 2.0])
    np.testing.assert_allclose(df["Rrs_555_mean"], [10.0, 20.0])


def test_write_mission_database_netcdf_handles_empty_mission(tmp_path):
    db = MissionDatabase(mission="Empty", waves=WAVES, stations=[])
    path = tmp_path / "empty.nc"

    write_mission_database_netcdf(db, path)
    reloaded = xr.open_dataset(path)
    try:
        assert reloaded.sizes["station"] == 0
    finally:
        reloaded.close()
