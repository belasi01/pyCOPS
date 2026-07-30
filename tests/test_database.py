from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycops.processing.database import (
    STANDARD_WAVELENGTHS,
    aggregate_station,
    assemble_mission_database,
    build_mission_database,
)

WAVES = np.array([443.0, 555.0])


def _write_fake_nc(
    path,
    *,
    rrs,
    ed0,
    waves=WAVES,
    method="Rrs.0p",
    fu=5.0,
    chl_flag=None,
    sun_zenith=None,
    lon=None,
    lat=None,
    time=None,
    par_0=None,
    kd_par_1pct=None,
    kd_par_10pct=None,
    kd_par_pd=None,
):
    """A minimal, hand-built .nc matching just the fields aggregate_station() reads -- avoids
    needing a full synthetic radiometric profile through process_cast() for every test case,
    and lets each test assert exact, hand-computed mean/sd values."""
    time = time if time is not None else pd.date_range("2019-08-17T12:00:00", periods=3, freq="s")
    ds = xr.Dataset(
        {
            "rrs_0p_recommended": ("wavelength", np.asarray(rrs, dtype=float)),
            "ed0_value_at_0": ("wavelength", np.asarray(ed0, dtype=float)),
        },
        coords={"wavelength": waves, "time": time},
    )
    fu_label = "linear" if method == "Rrs.0p.linear" else "loess"
    ds.attrs["rrs_method"] = method
    ds.attrs[f"qwip_{fu_label}_fu"] = fu
    ds.attrs["sun_zenith_deg"] = sun_zenith if sun_zenith is not None else float("nan")
    ds.attrs["longitude"] = lon if lon is not None else float("nan")
    ds.attrs["latitude"] = lat if lat is not None else float("nan")
    ds.attrs["chl_flag"] = chl_flag if chl_flag is not None else float("nan")
    if par_0 is not None:
        ds.attrs["par_0"] = par_0
    if kd_par_1pct is not None:
        ds.attrs["kd_par_1pct"] = kd_par_1pct
    if kd_par_10pct is not None:
        ds.attrs["kd_par_10pct"] = kd_par_10pct
    if kd_par_pd is not None:
        ds.attrs["kd_par_pd"] = kd_par_pd
    ds.to_netcdf(path, engine="netcdf4")


def _make_station(directory, cast_specs, select_rows=None):
    """cast_specs: {stem: dict(kwargs for _write_fake_nc)}. select_rows, if given, is the raw
    select.cops.dat text -- otherwise every cast is left kept-by-default (no select.cops.dat)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "init.cops.dat").write_text("")  # find_deployment_folders only checks existence
    nc_dir = directory / "nc"
    nc_dir.mkdir()
    for stem, kwargs in cast_specs.items():
        _write_fake_nc(nc_dir / f"{stem}.nc", **kwargs)
    if select_rows is not None:
        (directory / "select.cops.dat").write_text(select_rows)


def test_aggregate_station_mean_sd_matches_manual_computation(tmp_path):
    directory = tmp_path / "20200101_StationABC" / "cops"
    _make_station(
        directory,
        {
            "CAST_001": dict(rrs=[1.0, 10.0], ed0=[100.0, 200.0]),
            "CAST_002": dict(rrs=[2.0, 20.0], ed0=[110.0, 210.0]),
        },
    )

    result = aggregate_station(directory)

    i443 = list(STANDARD_WAVELENGTHS).index(443)
    i555 = list(STANDARD_WAVELENGTHS).index(555)
    assert result.n_casts == 2
    np.testing.assert_allclose(result.rrs.mean[[i443, i555]], [1.5, 15.0])
    np.testing.assert_allclose(result.rrs.sd[[i443, i555]], [np.std([1.0, 2.0], ddof=1), np.std([10.0, 20.0], ddof=1)])
    np.testing.assert_allclose(result.ed0_0p.mean[[i443, i555]], [105.0, 205.0])
    assert result.station_id == "ABC"


def test_aggregate_station_computes_scalar_par_kd_par_mean_sd(tmp_path):
    directory = tmp_path / "20200101_StationPAR" / "cops"
    _make_station(
        directory,
        {
            "CAST_001": dict(
                rrs=[1.0, 10.0], ed0=[100.0, 200.0], par_0=500.0, kd_par_1pct=0.5, kd_par_10pct=0.6, kd_par_pd=0.55
            ),
            "CAST_002": dict(
                rrs=[2.0, 20.0], ed0=[110.0, 210.0], par_0=520.0, kd_par_1pct=0.7, kd_par_10pct=0.8, kd_par_pd=0.75
            ),
        },
    )

    result = aggregate_station(directory)

    assert result.par_0.mean == pytest.approx(510.0)
    assert result.par_0.sd == pytest.approx(np.std([500.0, 520.0], ddof=1))
    assert result.kd_par_1pct.mean == pytest.approx(0.6)
    assert result.kd_par_10pct.mean == pytest.approx(0.7)
    assert result.kd_par_pd.mean == pytest.approx(0.65)


def test_aggregate_station_scalar_par_nan_when_no_cast_has_it(tmp_path):
    directory = tmp_path / "20200101_StationNoPAR" / "cops"
    _make_station(
        directory,
        {"CAST_001": dict(rrs=[1.0, 10.0], ed0=[100.0, 200.0])},
    )

    result = aggregate_station(directory)

    assert np.isnan(result.par_0.mean)
    assert np.isnan(result.par_0.sd)


def test_aggregate_station_excludes_rejected_cast(tmp_path):
    directory = tmp_path / "20200101_StationXYZ" / "cops"
    _make_station(
        directory,
        {
            "CAST_001": dict(rrs=[1.0, 1.0], ed0=[100.0, 100.0]),
            "CAST_002": dict(rrs=[99.0, 99.0], ed0=[999.0, 999.0]),
        },
        select_rows="CAST_001.csv;1;Rrs.0p;NA\nCAST_002.csv;0;Rrs.0p;NA\n",
    )

    result = aggregate_station(directory)

    assert result.n_casts == 1
    i443 = list(STANDARD_WAVELENGTHS).index(443)
    assert result.rrs.mean[i443] == 1.0


def test_aggregate_station_raises_when_no_nc_folder(tmp_path):
    directory = tmp_path / "20200101_StationEmpty" / "cops"
    directory.mkdir(parents=True)

    with pytest.raises(ValueError, match="no nc/"):
        aggregate_station(directory)


def test_aggregate_station_raises_when_zero_kept_casts(tmp_path):
    directory = tmp_path / "20200101_StationAllRejected" / "cops"
    _make_station(
        directory,
        {"CAST_001": dict(rrs=[1.0, 1.0], ed0=[100.0, 100.0])},
        select_rows="CAST_001.csv;0;Rrs.0p;NA\n",
    )

    with pytest.raises(ValueError, match="no kept casts"):
        aggregate_station(directory)


def test_aggregate_station_wavelength_tolerance(tmp_path):
    directory = tmp_path / "20200101_StationTol" / "cops"
    # 444.5 nm is within the 2 nm tolerance of the standard 443 nm band; 450 nm is not.
    _make_station(
        directory,
        {"CAST_001": dict(rrs=[7.0, 8.0], ed0=[100.0, 100.0], waves=np.array([444.5, 450.0]))},
    )

    result = aggregate_station(directory)

    i443 = list(STANDARD_WAVELENGTHS).index(443)
    assert result.rrs.mean[i443] == 7.0
    assert not np.isnan(result.rrs.mean[i443])


def test_aggregate_station_shadow_correction_method_from_chl_flag(tmp_path):
    directory = tmp_path / "20200101_StationChl" / "cops"
    _make_station(directory, {"CAST_001": dict(rrs=[1.0, 1.0], ed0=[100.0, 100.0], chl_flag=999.0)})

    result = aggregate_station(directory)

    assert result.shadow_correction_method == "abs.Kd.method"


def test_build_mission_database_skips_bad_station_without_aborting(tmp_path):
    good = tmp_path / "20200101_StationGood" / "cops"
    _make_station(good, {"CAST_001": dict(rrs=[1.0, 1.0], ed0=[100.0, 100.0])})
    bad = tmp_path / "20200102_StationBad" / "cops"
    bad.mkdir(parents=True)
    (bad / "init.cops.dat").write_text("")  # discovered, but has no nc/ folder -> skipped

    db = build_mission_database(tmp_path, mission="TestMission")

    assert len(db.stations) == 1
    assert db.stations[0].station_id == "Good"
    assert len(db.skipped) == 1
    assert db.skipped[0][0] == bad


def test_build_mission_database_drops_wavelengths_missing_everywhere(tmp_path):
    directory = tmp_path / "20200101_StationOneBand" / "cops"
    # Only 443 nm has real data anywhere in this (single-station) mission -- 555 nm should be
    # dropped from the shared wavelength grid, matching generate.cops.DB.R's ix.to.remove step.
    _make_station(directory, {"CAST_001": dict(rrs=[1.0, np.nan], ed0=[100.0, np.nan])})

    db = build_mission_database(tmp_path, mission="TestMission")

    assert 555.0 not in db.waves
    assert 443.0 in db.waves
    assert db.stations[0].rrs.mean.shape == db.waves.shape


def test_assemble_mission_database_used_by_ui_for_a_filtered_station_subset(tmp_path):
    """The UI tab checks/unchecks discovered stations before generating -- it calls
    aggregate_station() per checked folder itself, then assemble_mission_database() to apply the
    same wavelength-trimming build_mission_database() does, rather than re-discovering the whole
    parent folder (which would ignore the unchecked stations)."""
    directory = tmp_path / "20200101_StationOnlyOneChecked" / "cops"
    _make_station(directory, {"CAST_001": dict(rrs=[3.0, 4.0], ed0=[100.0, 100.0])})
    station = aggregate_station(directory)

    db = assemble_mission_database("TestMission", [station], skipped=[(tmp_path / "unchecked", "not selected")])

    assert db.mission == "TestMission"
    assert len(db.stations) == 1
    assert db.stations[0].station_id == "OnlyOneChecked"
    assert db.skipped == [(tmp_path / "unchecked", "not selected")]
