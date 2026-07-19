from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from pycops.io.netcdf import cast_result_to_dataset, write_cast_result
from pycops.processing.position import PositionOverride
from pycops.processing.process_cast import process_cast

WAVES = (340.0, 380.0, 443.0, 555.0)
K_TRUE = (2.0, 0.9, 0.3, 0.1)
LUZ_X0_TRUE = (0.0015, 0.006, 0.05, 0.3)
EDZ_X0_TRUE = (50.0, 80.0, 150.0, 300.0)
DELTA_CAPTEUR_LUZ = 0.238
DELTA_CAPTEUR_EDZ = -0.05


def _make_dataset(n=300, ed0_level=100.0, include_edz=True):
    sensor_depth = np.linspace(0.05, 6.0, n)
    waves = np.array(WAVES)
    K = np.array(K_TRUE)

    luz_true_depth = sensor_depth + DELTA_CAPTEUR_LUZ
    luz = np.array(LUZ_X0_TRUE)[None, :] * np.exp(-K[None, :] * luz_true_depth[:, None])

    ed0 = np.full((n, len(waves)), ed0_level)
    zeros = np.zeros(n)

    data_vars = {
        "Ed0": (("time", "wavelength"), ed0),
        "LuZ": (("time", "wavelength"), luz),
        "Ed0_Roll": ("time", zeros),
        "Ed0_Pitch": ("time", zeros),
        "EdZ_Roll": ("time", zeros),
        "EdZ_Pitch": ("time", zeros),
        "LuZ_Depth": ("time", sensor_depth),
        "LuZ_Temp": ("time", np.full(n, 10.0)),
    }
    if include_edz:
        edz_true_depth = sensor_depth + DELTA_CAPTEUR_EDZ
        data_vars["EdZ"] = (
            ("time", "wavelength"),
            np.array(EDZ_X0_TRUE)[None, :] * np.exp(-K[None, :] * edz_true_depth[:, None]),
        )

    times = pd.date_range("2019-08-18T18:18:00", periods=n, freq="s")
    return xr.Dataset(data_vars, coords={"time": times, "wavelength": waves})


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
    }


def _cast_result_with_shadow():
    ds = _make_dataset()
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.108833
    ds.attrs["latitude"] = 49.13445
    ds.attrs["qc_flag"] = 1
    return ds, process_cast(ds, _make_init())


def test_cast_result_to_dataset_has_expected_dims_and_coords():
    ds, result = _cast_result_with_shadow()
    out = cast_result_to_dataset(result, ds=ds)

    assert "wavelength" in out.coords
    assert "time" in out.coords
    assert "EdZ_depth" in out.coords
    assert "LuZ_depth" in out.coords
    np.testing.assert_allclose(out["wavelength"].values, result.waves)
    np.testing.assert_array_equal(out["time"].values, ds["time"].values)


def test_cast_result_to_dataset_matches_source_arrays():
    ds, result = _cast_result_with_shadow()
    out = cast_result_to_dataset(result, ds=ds)

    np.testing.assert_allclose(out["LuZ_fitted"].values, result.instrument_fits["LuZ"].aop_fitted)
    np.testing.assert_allclose(out["LuZ_value_at_0"].values, result.instrument_fits["LuZ"].value_at_0)
    np.testing.assert_allclose(
        out["LuZ_surface_value_at_surface"].values, result.instrument_fits["LuZ"].surface_linear.value_at_surface
    )
    np.testing.assert_allclose(out["rrs_0p_linear"].values, result.rrs_linear.rrs_0p, equal_nan=True)
    np.testing.assert_allclose(out["rrs_0p_recommended"].values, result.recommended_rrs.rrs_0p, equal_nan=True)


def test_cast_result_to_dataset_KZ_K0_padded_to_depth_grid_length():
    ds, result = _cast_result_with_shadow()
    out = cast_result_to_dataset(result, ds=ds)

    luz_fit = result.instrument_fits["LuZ"]
    assert out["LuZ_KZ"].shape == (len(luz_fit.depth_grid), len(result.waves))
    assert np.all(np.isnan(out["LuZ_KZ"].values[0]))
    np.testing.assert_allclose(out["LuZ_KZ"].values[1:], luz_fit.KZ)


def test_cast_result_to_dataset_includes_shadow_correction_when_present():
    ds, result = _cast_result_with_shadow()
    out = cast_result_to_dataset(result, ds=ds)

    assert "LuZ" in result.shadow_corrections
    np.testing.assert_allclose(out["LuZ_shadow_correction"].values, result.shadow_corrections["LuZ"].correction)
    assert out.attrs["LuZ_absorption_source"] == result.shadow_corrections["LuZ"].absorption.source


def test_cast_result_to_dataset_carries_over_cast_attrs():
    ds, result = _cast_result_with_shadow()
    out = cast_result_to_dataset(result, ds=ds)

    assert out.attrs["chl_flag"] == 999.0
    assert out.attrs["longitude"] == -68.108833
    assert out.attrs["latitude"] == 49.13445
    assert out.attrs["qc_flag"] == 1


def test_cast_result_to_dataset_without_ds_still_works():
    ds = _make_dataset()
    result = process_cast(ds, _make_init())  # no chl/position attrs -- shadow correction skipped

    out = cast_result_to_dataset(result)  # no ds passed

    assert "time" in out.coords
    assert out["time"].shape == (300,)
    assert "chl_flag" not in out.attrs


def test_write_cast_result_round_trips_through_netcdf(tmp_path):
    ds, result = _cast_result_with_shadow()
    path = tmp_path / "cast.nc"

    write_cast_result(result, path, ds=ds)
    reloaded = xr.open_dataset(path)
    try:
        np.testing.assert_allclose(reloaded["rrs_0p_linear"].values, result.rrs_linear.rrs_0p, equal_nan=True)
        np.testing.assert_allclose(reloaded["LuZ_value_at_0"].values, result.instrument_fits["LuZ"].value_at_0)
        assert reloaded.attrs["chl_flag"] == 999.0
    finally:
        reloaded.close()


def test_cast_result_to_dataset_no_luz_still_builds(tmp_path):
    ds = _make_dataset(include_edz=True)
    ds = ds.drop_vars("LuZ")
    result = process_cast(ds, _make_init())

    out = cast_result_to_dataset(result, ds=ds)

    assert "LuZ_fitted" not in out.data_vars
    assert "EdZ_fitted" in out.data_vars
    assert "rrs_0p_linear" not in out.data_vars


def test_cast_result_to_dataset_prefers_resolved_position_over_stale_attrs():
    # info.cops.dat said one thing, but a position_override (or a GPS file, via
    # process_deployment()) resolved a different position -- the persisted
    # file must reflect what was actually used, not the stale ds.attrs value.
    ds = _make_dataset()
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.108833
    ds.attrs["latitude"] = 49.13445

    result = process_cast(ds, _make_init(), position_override=PositionOverride(longitude=10.0, latitude=-30.0))
    out = cast_result_to_dataset(result, ds=ds)

    assert out.attrs["longitude"] == 10.0
    assert out.attrs["latitude"] == -30.0


def test_cast_result_to_dataset_longitude_nan_when_unresolved():
    ds = _make_dataset()  # no chl/position attrs -- shadow correction never resolves a position
    result = process_cast(ds, _make_init())

    out = cast_result_to_dataset(result, ds=ds)

    assert np.isnan(out.attrs["longitude"])
    assert np.isnan(out.attrs["latitude"])
