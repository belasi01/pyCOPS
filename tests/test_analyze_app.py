from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from conftest import _write_cast_file  # noqa: E402
from pycops.io.netcdf import write_cast_result  # noqa: E402
from pycops.processing.process_cast import process_cast  # noqa: E402

_APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "pycops" / "ui" / "clean_app.py")

_CAST_FILE = "WISE_CAST_001_190817_220856_URC.csv"
_CAST_STEM = "WISE_CAST_001_190817_220856_URC"

WAVES = (340.0, 380.0, 443.0, 555.0)
K_TRUE = (2.0, 0.9, 0.3, 0.1)
LUZ_X0 = (0.0015, 0.006, 0.05, 0.3)
EDZ_X0 = (50.0, 80.0, 150.0, 300.0)
EUZ_X0 = (0.002, 0.008, 0.06, 0.35)
DELTA_LUZ = 0.238
DELTA_EDZ = -0.05
DELTA_EUZ = 0.238


def _make_full_dataset(n=300):
    """EdZ+LuZ+EuZ, a real position/chl (shadow correction) and shallow=True (bottom
    reflectance) -- exercises every optional section render_analyze_tab() can show."""
    depth = np.linspace(0.05, 8.0, n)
    waves = np.array(WAVES)
    K = np.array(K_TRUE)
    luz = np.array(LUZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_LUZ)[:, None])
    edz = np.array(EDZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_EDZ)[:, None])
    euz = np.array(EUZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_EUZ)[:, None])
    ed0 = np.full((n, len(waves)), 100.0)
    zeros = np.zeros(n)
    times = pd.date_range("2019-08-17T22:08:56", periods=n, freq="s")
    ds = xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), ed0),
            "LuZ": (("time", "wavelength"), luz),
            "EdZ": (("time", "wavelength"), edz),
            "EuZ": (("time", "wavelength"), euz),
            "Ed0_Roll": ("time", zeros),
            "Ed0_Pitch": ("time", zeros),
            "EdZ_Roll": ("time", zeros),
            "EdZ_Pitch": ("time", zeros),
            "EuZ_Roll": ("time", zeros),
            "EuZ_Pitch": ("time", zeros),
            "LuZ_Depth": ("time", depth),
            "EuZ_Depth": ("time", depth),
            "LuZ_Temp": ("time", np.full(n, 10.0)),
            "EuZ_Temp": ("time", np.full(n, 10.0)),
        },
        coords={"time": times, "wavelength": waves},
    )
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.11626
    ds.attrs["latitude"] = 49.24872
    ds.attrs["shallow"] = True
    return ds


def _make_full_init():
    nan = float("nan")
    return {
        "depth.is.on": "LuZ",
        "indice.water": 1.34,
        "rau.Fresnel": 0.043,
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 7.0, "LuZ": 7.0, "EuZ": 7.0},
        "radius.instrument.optics": {"Ed0": 0.035, "EdZ": 0.035, "LuZ": 0.035, "EuZ": 0.035},
        "delta.capteur.optics": {"Ed0": 0.0, "EdZ": DELTA_EDZ, "LuZ": DELTA_LUZ, "EuZ": DELTA_EUZ},
        "sub.surface.removed.layer.optics": {"Ed0": 0.0, "EdZ": 0.3, "LuZ": 0.0, "EuZ": 0.0},
        "depth.interval.for.smoothing.optics": {"Ed0": 10.0, "EdZ": 3.0, "LuZ": 3.0, "EuZ": 3.0},
        "linear.fit.Rsquared.threshold.optics": {"Ed0": nan, "EdZ": 0.5, "LuZ": 0.5, "EuZ": 0.5},
        "linear.fit.max.delta.depth.optics": {"Ed0": nan, "EdZ": 3.0, "LuZ": 2.5, "EuZ": 2.5},
        "depth.discretization": [0, 0.01, 1, 0.02, 2, 0.05, 5, 0.1, 10, 0.2, 20, 0.5, 50, 1, 100, 2, 200, 5, 500],
        "bandwidth": 10.0,
    }


def _make_minimal_dataset(n=200):
    """LuZ+EdZ only, no position/chl (no shadow correction), not shallow (no bottom)."""
    depth = np.linspace(0.05, 5.0, n)
    waves = np.array(WAVES)
    K = np.array(K_TRUE)
    luz = np.array(LUZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_LUZ)[:, None])
    edz = np.array(EDZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_EDZ)[:, None])
    ed0 = np.full((n, len(waves)), 100.0)
    zeros = np.zeros(n)
    times = pd.date_range("2019-08-17T22:08:56", periods=n, freq="s")
    return xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), ed0),
            "LuZ": (("time", "wavelength"), luz),
            "EdZ": (("time", "wavelength"), edz),
            "Ed0_Roll": ("time", zeros),
            "Ed0_Pitch": ("time", zeros),
            "EdZ_Roll": ("time", zeros),
            "EdZ_Pitch": ("time", zeros),
            "LuZ_Depth": ("time", depth),
            "LuZ_Temp": ("time", np.full(n, 10.0)),
        },
        coords={"time": times, "wavelength": waves},
    )


def _make_minimal_init():
    nan = float("nan")
    return {
        "depth.is.on": "LuZ",
        "indice.water": 1.34,
        "rau.Fresnel": 0.043,
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 7.0, "LuZ": 7.0},
        "radius.instrument.optics": {"Ed0": 0.035, "EdZ": 0.035, "LuZ": 0.035},
        "delta.capteur.optics": {"Ed0": 0.0, "EdZ": DELTA_EDZ, "LuZ": DELTA_LUZ},
        "sub.surface.removed.layer.optics": {"Ed0": 0.0, "EdZ": 0.3, "LuZ": 0.0},
        "depth.interval.for.smoothing.optics": {"Ed0": 10.0, "EdZ": 3.0, "LuZ": 3.0},
        "linear.fit.Rsquared.threshold.optics": {"Ed0": nan, "EdZ": 0.5, "LuZ": 0.5},
        "linear.fit.max.delta.depth.optics": {"Ed0": nan, "EdZ": 3.0, "LuZ": 2.5},
        "depth.discretization": [0, 0.01, 1, 0.02, 2, 0.05, 5, 0.1, 10, 0.2, 20, 0.5, 50, 1, 100, 2, 200, 5, 500],
        "bandwidth": 10.0,
    }


def _write_nc(tmp_path, ds, init, filename=_CAST_FILE, stem=_CAST_STEM):
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir(parents=True, exist_ok=True)
    result = process_cast(ds, init)
    write_cast_result(result, nc_dir / f"{stem}.nc", ds=ds)
    return result


def test_analyze_tab_missing_nc_folder_shows_error(tmp_path):
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert any("nc/ subfolder" in e.value for e in at.error)


def test_analyze_tab_full_cast_renders_every_section(tmp_path):
    _write_nc(tmp_path, _make_full_dataset(), _make_full_init())
    _write_cast_file(tmp_path, _CAST_FILE)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=60)

    assert not at.exception
    assert at.selectbox(key="analyze_cast_select").options == [_CAST_STEM]

    # Overview + Ed0 stability render unconditionally.
    assert any("Overview" in s.value for s in at.subheader)
    assert any("Ed0 stability" in s.value for s in at.subheader)

    # All three depth-profiled instruments got their own expanders.
    expander_labels = [e.label for e in at.expander]
    for instrument in ("EdZ", "LuZ", "EuZ"):
        assert f"{instrument} depth profile" in expander_labels
        assert f"{instrument} attenuation (K)" in expander_labels

    # Exercise the wavelength drill-down (raw-scan overlay code path) on LuZ.
    at.selectbox(key="analyze_LuZ_depth_wave").set_value("340").run(timeout=30)
    assert not at.exception

    assert any("Rrs / Lw / nLw spectra" in s.value for s in at.subheader)
    assert "LuZ shadow correction" in expander_labels
    assert "EuZ shadow correction" in expander_labels
    assert any("QWIP" in s.value for s in at.subheader)
    assert "LuZ bottom reflectance" in expander_labels
    assert "EuZ bottom reflectance" in expander_labels


def test_analyze_tab_minimal_cast_skips_optional_sections(tmp_path):
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    expander_labels = [e.label for e in at.expander]
    assert "EuZ depth profile" not in expander_labels
    assert "LuZ shadow correction" not in expander_labels
    assert "LuZ bottom reflectance" not in expander_labels
    # QWIP still renders even without shadow correction -- Rrs falls back to the uncorrected
    # LuZ surface value rather than being skipped entirely, so QWIP still has something to score.


def test_analyze_tab_missing_raw_file_still_renders_nc_content(tmp_path):
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    # No raw cast file written alongside nc/.

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert any("raw cast file not found" in c.value for c in at.caption)
    assert any("Overview" in s.value for s in at.subheader)
