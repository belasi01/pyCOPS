from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pycops.processing.deployment as deployment_module
from pycops.io.config import CastInfo
from pycops.io.discovery import CastRecord, CastReadFailure, CastSelection, Deployment, DeploymentCastsResult
from pycops.processing.deployment import process_deployment
from pycops.processing.position import PositionOverride

GPS_TSV_FOR_PROFILE = (
    '"[DateTime]"\t"DateTimeUTC"\t"Millisecond"\t"[GpsTime]"\t"Latitude"\t"Longitude"\t"SatelliteCount"\n'
    '"6/5/2018 7:50:00 PM"\t"06/05/2018 07:50:00 PM"\t0\t"6/5/2018 7:49:59 PM"\t63.1770\t-81.8483\t11\n'
    '"6/5/2018 7:51:00 PM"\t"06/05/2018 07:51:00 PM"\t0\t"6/5/2018 7:50:59 PM"\t63.1772\t-81.8481\t9\n'
)

WAVES = (340.0, 380.0, 443.0, 555.0)
K_TRUE = (2.0, 0.9, 0.3, 0.1)
LUZ_X0_TRUE = (0.0015, 0.006, 0.05, 0.3)
EDZ_X0_TRUE = (50.0, 80.0, 150.0, 300.0)
DELTA_CAPTEUR_LUZ = 0.238
DELTA_CAPTEUR_EDZ = -0.05

PROFILE_CAST = "hudsonbay_CAST_001_180605_194923_URC.csv"
BIOSHADE_CAST = "hudsonbay_SB_180605_192518_URC.csv"


def _make_profile_dataset(n=300, ed0_level=100.0):
    sensor_depth = np.linspace(0.05, 6.0, n)
    waves = np.array(WAVES)
    K = np.array(K_TRUE)

    luz_true_depth = sensor_depth + DELTA_CAPTEUR_LUZ
    luz = np.array(LUZ_X0_TRUE)[None, :] * np.exp(-K[None, :] * luz_true_depth[:, None])
    edz_true_depth = sensor_depth + DELTA_CAPTEUR_EDZ
    edz = np.array(EDZ_X0_TRUE)[None, :] * np.exp(-K[None, :] * edz_true_depth[:, None])

    ed0 = np.full((n, len(waves)), ed0_level)
    zeros = np.zeros(n)

    return xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), ed0),
            "LuZ": (("time", "wavelength"), luz),
            "EdZ": (("time", "wavelength"), edz),
            "Ed0_Roll": ("time", zeros),
            "Ed0_Pitch": ("time", zeros),
            "EdZ_Roll": ("time", zeros),
            "EdZ_Pitch": ("time", zeros),
            "LuZ_Depth": ("time", sensor_depth),
            "LuZ_Temp": ("time", np.full(n, 10.0)),
        },
        coords={"time": np.arange(n), "wavelength": waves},
    )


def _with_real_time(ds):
    times = pd.date_range("2018-06-05T19:49:23", periods=ds.sizes["time"], freq="s")
    return ds.assign_coords(time=times)


def _make_bioshade_dataset(n=300, occlusion_index=150, position_period_s=60.0):
    time_s = np.linspace(0, n - 1, n)
    times = np.datetime64("2018-06-05T19:25:18") + (time_s * 1e9).astype("timedelta64[ns]")
    position = (time_s % position_period_s) / position_period_s * 26000.0

    waves = np.array(WAVES)
    ed0_tot = np.array([80.0, 70.0, 50.0, 25.0])
    ed0 = np.tile(ed0_tot, (n, 1))
    ed0[occlusion_index, :] = ed0_tot * 0.4  # sun-occlusion scan

    zeros = np.zeros(n)
    return xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), ed0),
            "Ed0_Roll": ("time", zeros),
            "Ed0_Pitch": ("time", zeros),
            "BioShadePosition": ("time", position),
        },
        coords={"time": times, "wavelength": waves},
    )


def _make_init():
    nan = float("nan")
    return {
        "depth.is.on": "LuZ",
        "indice.water": 1.34,
        "rau.Fresnel": 0.043,
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 7.0, "LuZ": 7.0, "EuZ": 7.0},
        "radius.instrument.optics": {"Ed0": 0.035, "EdZ": 0.035, "LuZ": 0.035, "EuZ": 0.035},
        "delta.capteur.optics": {"Ed0": 0.0, "EdZ": DELTA_CAPTEUR_EDZ, "LuZ": DELTA_CAPTEUR_LUZ, "EuZ": 0.238},
        "sub.surface.removed.layer.optics": {"Ed0": 0.0, "EdZ": 0.3, "LuZ": 0.0, "EuZ": 0.0},
        "depth.interval.for.smoothing.optics": {"Ed0": 10.0, "EdZ": 3.0, "LuZ": 3.0, "EuZ": 3.0},
        "linear.fit.Rsquared.threshold.optics": {"Ed0": nan, "EdZ": 0.5, "LuZ": 0.5, "EuZ": 0.5},
        "linear.fit.max.delta.depth.optics": {"Ed0": nan, "EdZ": 3.0, "LuZ": 2.5, "EuZ": 2.5},
        "depth.discretization": [0, 0.01, 1, 0.02, 2, 0.05, 5, 0.1, 10, 0.2, 20, 0.5, 50, 1, 100, 2, 200, 5, 500],
        "time.window": [0.0, 10000.0],
    }


def _cast_info(file, chl_flag):
    return CastInfo(
        file=file,
        longitude=-81.8482666651408,
        latitude=63.1770166714986,
        chl_flag=chl_flag,
        time_window=None,
        sub_surface_removed_layer=None,
        tiltmax=None,
        depth_interval_for_smoothing=None,
        dark_files=[],
    )


def _make_deployment(tmp_path, profile_chl_flag=999.0):
    directory = tmp_path
    casts = [
        CastRecord(
            path=directory / PROFILE_CAST,
            info=_cast_info(PROFILE_CAST, profile_chl_flag),
            selection=CastSelection(file=PROFILE_CAST, flag=1, method="Rrs.0p.linear", extra="NA"),
        ),
        CastRecord(
            path=directory / BIOSHADE_CAST,
            info=_cast_info(BIOSHADE_CAST, None),
            selection=CastSelection(file=BIOSHADE_CAST, flag=2, method="Rrs.0p.linear", extra="NA"),
        ),
    ]
    return Deployment(directory=directory, init=_make_init(), casts=casts)


def _with_position_attrs(ds, chl_flag):
    # mirrors what the real read_deployment_casts() sets from info.cops.dat --
    # these tests bypass it via monkeypatch, so they must set attrs themselves.
    ds.attrs["chl_flag"] = chl_flag
    ds.attrs["longitude"] = -81.8482666651408
    ds.attrs["latitude"] = 63.1770166714986
    return ds


def _with_chl_only_attrs(ds, chl_flag):
    # simulates info.cops.dat's longitude/latitude being NA for this cast.
    ds.attrs["chl_flag"] = chl_flag
    return ds


def _patch_discovery(monkeypatch, deployment, datasets, failures=None):
    monkeypatch.setattr(deployment_module, "discover_deployment", lambda directory: deployment)
    monkeypatch.setattr(
        deployment_module,
        "read_deployment_casts",
        lambda dep, only_kept=True: DeploymentCastsResult(datasets=datasets, failures=failures or []),
    )


def test_process_deployment_finds_and_uses_bioshade(tmp_path, monkeypatch):
    deployment = _make_deployment(tmp_path)
    datasets = {
        PROFILE_CAST: _with_position_attrs(_with_real_time(_make_profile_dataset()), 999.0),
        BIOSHADE_CAST: _make_bioshade_dataset(),
    }
    _patch_discovery(monkeypatch, deployment, datasets)

    result = process_deployment(tmp_path)

    assert set(result.bioshade_results) == {BIOSHADE_CAST}
    assert result.bioshade_used is result.bioshade_results[BIOSHADE_CAST]
    assert set(result.cast_results) == {PROFILE_CAST}
    assert result.processing_failures == []
    assert result.read_failures == []

    cast_result = result.cast_results[PROFILE_CAST]
    assert cast_result.shadow_correction_note is None
    np.testing.assert_allclose(
        cast_result.shadow_corrections["LuZ"].edif,
        np.interp(np.array(WAVES), result.bioshade_used.waves, result.bioshade_used.ed0_dif),
    )


def test_process_deployment_matches_manual_process_cast_with_bioshade(tmp_path, monkeypatch):
    from pycops.processing.bioshade import process_bioshade
    from pycops.processing.process_cast import process_cast

    deployment = _make_deployment(tmp_path)
    profile_ds = _with_position_attrs(_with_real_time(_make_profile_dataset()), 999.0)
    bioshade_ds = _make_bioshade_dataset()
    datasets = {PROFILE_CAST: profile_ds, BIOSHADE_CAST: bioshade_ds}
    _patch_discovery(monkeypatch, deployment, datasets)

    result = process_deployment(tmp_path)

    expected_bioshade = process_bioshade(bioshade_ds, deployment.init)
    expected_cast = process_cast(profile_ds, deployment.init, bioshade=expected_bioshade)

    np.testing.assert_allclose(
        result.cast_results[PROFILE_CAST].rrs_linear.rrs_0p, expected_cast.rrs_linear.rrs_0p, equal_nan=True
    )


def test_process_deployment_without_bioshade_cast_falls_back_to_gregg_carder(tmp_path, monkeypatch):
    directory = tmp_path
    casts = [
        CastRecord(
            path=directory / PROFILE_CAST,
            info=_cast_info(PROFILE_CAST, 999.0),
            selection=CastSelection(file=PROFILE_CAST, flag=1, method="Rrs.0p.linear", extra="NA"),
        )
    ]
    deployment = Deployment(directory=directory, init=_make_init(), casts=casts)
    datasets = {PROFILE_CAST: _with_position_attrs(_with_real_time(_make_profile_dataset()), 999.0)}
    _patch_discovery(monkeypatch, deployment, datasets)

    result = process_deployment(tmp_path)

    assert result.bioshade_results == {}
    assert result.bioshade_used is None
    assert result.cast_results[PROFILE_CAST].shadow_correction_note is None


def test_process_deployment_uses_absorption_table_for_chl_zero(tmp_path, monkeypatch):
    deployment = _make_deployment(tmp_path, profile_chl_flag=0.0)
    (tmp_path / "absorption.cops.dat").write_text(
        "file;340;380;443;555\n" f"{PROFILE_CAST};5.0;3.0;1.0;0.5\n"
    )
    datasets = {
        PROFILE_CAST: _with_position_attrs(_with_real_time(_make_profile_dataset()), 0.0),
        BIOSHADE_CAST: _make_bioshade_dataset(),
    }
    _patch_discovery(monkeypatch, deployment, datasets)

    result = process_deployment(tmp_path)

    cast_result = result.cast_results[PROFILE_CAST]
    assert cast_result.shadow_correction_note is None
    assert cast_result.shadow_corrections["LuZ"].absorption.source == "file"
    np.testing.assert_allclose(cast_result.shadow_corrections["LuZ"].absorption.values, [5.0, 3.0, 1.0, 0.5])


def test_process_deployment_isolates_one_bad_cast(tmp_path, monkeypatch):
    deployment = _make_deployment(tmp_path)
    broken_profile = _with_position_attrs(_with_real_time(_make_profile_dataset()), 999.0).drop_vars("Ed0")
    datasets = {PROFILE_CAST: broken_profile, BIOSHADE_CAST: _make_bioshade_dataset()}
    _patch_discovery(monkeypatch, deployment, datasets)

    with pytest.warns(UserWarning, match=PROFILE_CAST):
        result = process_deployment(tmp_path)

    assert PROFILE_CAST not in result.cast_results
    assert any(f.file == PROFILE_CAST for f in result.processing_failures)
    # the BioShade cast itself still processed fine
    assert BIOSHADE_CAST in result.bioshade_results


def test_process_deployment_propagates_read_failures(tmp_path, monkeypatch):
    deployment = _make_deployment(tmp_path)
    datasets = {BIOSHADE_CAST: _make_bioshade_dataset()}
    failure = CastReadFailure(file=PROFILE_CAST, path=tmp_path / PROFILE_CAST, error="ValueError: boom")
    _patch_discovery(monkeypatch, deployment, datasets, failures=[failure])

    result = process_deployment(tmp_path)

    assert result.read_failures == [failure]
    assert PROFILE_CAST not in result.cast_results


def test_process_deployment_uses_gps_file_when_info_lacks_position(tmp_path, monkeypatch):
    deployment = _make_deployment(tmp_path)
    profile_ds = _with_chl_only_attrs(_with_real_time(_make_profile_dataset()), 999.0)
    datasets = {PROFILE_CAST: profile_ds, BIOSHADE_CAST: _make_bioshade_dataset()}
    _patch_discovery(monkeypatch, deployment, datasets)
    (tmp_path / "GPS_180605.tsv").write_text(GPS_TSV_FOR_PROFILE)

    result = process_deployment(tmp_path)

    cast_result = result.cast_results[PROFILE_CAST]
    assert cast_result.shadow_correction_note is None
    assert "LuZ" in cast_result.shadow_corrections


def test_process_deployment_no_position_source_reports_note(tmp_path, monkeypatch):
    deployment = _make_deployment(tmp_path)
    profile_ds = _with_chl_only_attrs(_with_real_time(_make_profile_dataset()), 999.0)
    datasets = {PROFILE_CAST: profile_ds, BIOSHADE_CAST: _make_bioshade_dataset()}
    _patch_discovery(monkeypatch, deployment, datasets)
    # no GPS file, no manual override

    result = process_deployment(tmp_path)

    cast_result = result.cast_results[PROFILE_CAST]
    assert cast_result.shadow_corrections == {}
    assert "position" in cast_result.shadow_correction_note.lower()


def test_process_deployment_manual_override_wins_over_gps_file(tmp_path, monkeypatch):
    deployment = _make_deployment(tmp_path)
    profile_ds = _with_chl_only_attrs(_with_real_time(_make_profile_dataset()), 999.0)
    datasets = {PROFILE_CAST: profile_ds, BIOSHADE_CAST: _make_bioshade_dataset()}
    _patch_discovery(monkeypatch, deployment, datasets)
    (tmp_path / "GPS_180605.tsv").write_text(GPS_TSV_FOR_PROFILE)

    result_gps = process_deployment(tmp_path)
    result_override = process_deployment(
        tmp_path, position_overrides={PROFILE_CAST: PositionOverride(longitude=10.0, latitude=-40.0)}
    )

    gps_rrs = result_gps.cast_results[PROFILE_CAST].rrs_linear.rrs_0p
    override_rrs = result_override.cast_results[PROFILE_CAST].rrs_linear.rrs_0p
    finite = np.isfinite(gps_rrs) & np.isfinite(override_rrs)
    assert finite.any()
    assert not np.allclose(gps_rrs[finite], override_rrs[finite])


def test_process_deployment_manual_utc_time_override(tmp_path, monkeypatch):
    deployment = _make_deployment(tmp_path)
    profile_ds = _with_position_attrs(_with_real_time(_make_profile_dataset()), 999.0)
    datasets = {PROFILE_CAST: profile_ds, BIOSHADE_CAST: _make_bioshade_dataset()}
    _patch_discovery(monkeypatch, deployment, datasets)

    result_normal = process_deployment(tmp_path)
    result_shifted = process_deployment(
        tmp_path,
        position_overrides={PROFILE_CAST: PositionOverride(utc_time=pd.Timestamp("2018-01-05T04:49:23"))},
    )

    normal_rrs = result_normal.cast_results[PROFILE_CAST].rrs_linear.rrs_0p
    shifted_rrs = result_shifted.cast_results[PROFILE_CAST].rrs_linear.rrs_0p
    finite = np.isfinite(normal_rrs) & np.isfinite(shifted_rrs)
    assert finite.any()
    assert not np.allclose(normal_rrs[finite], shifted_rrs[finite])
