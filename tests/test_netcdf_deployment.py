from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from pycops.io.netcdf import write_deployment_result
from pycops.processing.deployment import DeploymentProcessingResult
from pycops.processing.process_cast import process_cast

WAVES = (340.0, 380.0, 443.0, 555.0)
K_TRUE = (2.0, 0.9, 0.3, 0.1)
LUZ_X0_TRUE = (0.0015, 0.006, 0.05, 0.3)
EDZ_X0_TRUE = (50.0, 80.0, 150.0, 300.0)
DELTA_CAPTEUR_LUZ = 0.238
DELTA_CAPTEUR_EDZ = -0.05

CAST_A = "hudsonbay_CAST_001_180605_194923_URC.csv"
CAST_B = "hudsonbay_CAST_002_180605_195752_URC.csv"


def _make_dataset(n=300, ed0_level=100.0):
    sensor_depth = np.linspace(0.05, 6.0, n)
    waves = np.array(WAVES)
    K = np.array(K_TRUE)

    luz_true_depth = sensor_depth + DELTA_CAPTEUR_LUZ
    luz = np.array(LUZ_X0_TRUE)[None, :] * np.exp(-K[None, :] * luz_true_depth[:, None])
    edz_true_depth = sensor_depth + DELTA_CAPTEUR_EDZ
    edz = np.array(EDZ_X0_TRUE)[None, :] * np.exp(-K[None, :] * edz_true_depth[:, None])

    ed0 = np.full((n, len(waves)), ed0_level)
    zeros = np.zeros(n)

    times = pd.date_range("2018-06-05T19:49:23", periods=n, freq="s")
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
    }


def _make_result_and_datasets():
    init = _make_init()
    datasets = {}
    cast_results = {}
    for i, file in enumerate((CAST_A, CAST_B)):
        ds = _make_dataset(ed0_level=100.0 + i)
        ds.attrs["chl_flag"] = 999.0
        ds.attrs["longitude"] = -81.848 + i * 0.001
        ds.attrs["latitude"] = 63.177
        datasets[file] = ds
        cast_results[file] = process_cast(ds, init)

    result = DeploymentProcessingResult(
        cast_results=cast_results,
        bioshade_results={},
        bioshade_used=None,
        read_failures=[],
        processing_failures=[],
    )
    return result, datasets


def test_write_deployment_result_writes_one_file_per_cast(tmp_path):
    result, datasets = _make_result_and_datasets()

    written = write_deployment_result(result, tmp_path, datasets=datasets)

    assert set(written) == {CAST_A, CAST_B}
    assert written[CAST_A] == tmp_path / "hudsonbay_CAST_001_180605_194923_URC.nc"
    assert written[CAST_A].exists()
    assert written[CAST_B].exists()


def test_write_deployment_result_files_carry_correct_per_cast_data(tmp_path):
    result, datasets = _make_result_and_datasets()

    written = write_deployment_result(result, tmp_path, datasets=datasets)

    for file in (CAST_A, CAST_B):
        reloaded = xr.open_dataset(written[file])
        try:
            np.testing.assert_allclose(
                reloaded["rrs_0p_linear"].values, result.cast_results[file].rrs_linear.rrs_0p, equal_nan=True
            )
            assert reloaded.attrs["longitude"] == datasets[file].attrs["longitude"]
        finally:
            reloaded.close()

    # the two casts' longitudes differ -- confirm they weren't accidentally cross-wired
    a = xr.open_dataset(written[CAST_A])
    b = xr.open_dataset(written[CAST_B])
    try:
        assert a.attrs["longitude"] != b.attrs["longitude"]
    finally:
        a.close()
        b.close()


def test_write_deployment_result_creates_target_directory(tmp_path):
    result, datasets = _make_result_and_datasets()
    target = tmp_path / "nested" / "output"

    written = write_deployment_result(result, target, datasets=datasets)

    assert target.is_dir()
    assert all(path.exists() for path in written.values())


def test_write_deployment_result_without_datasets_still_writes(tmp_path):
    result, _ = _make_result_and_datasets()

    written = write_deployment_result(result, tmp_path)

    assert len(written) == 2
    reloaded = xr.open_dataset(written[CAST_A])
    try:
        assert "time" in reloaded.coords
        assert "chl_flag" not in reloaded.attrs  # no ds -> no chl_flag/qc_flag carried over
    finally:
        reloaded.close()


def test_write_deployment_result_empty_cast_results(tmp_path):
    result = DeploymentProcessingResult(
        cast_results={}, bioshade_results={}, bioshade_used=None, read_failures=[], processing_failures=[]
    )

    written = write_deployment_result(result, tmp_path)

    assert written == {}
    assert tmp_path.is_dir()
