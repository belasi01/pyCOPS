from __future__ import annotations

import numpy as np
import xarray as xr

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
        "EdZ_Roll": ("time", zeros),  # LuZ has no inclinometer -> falls back to EdZ
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

    return xr.Dataset(data_vars, coords={"time": np.arange(n), "wavelength": waves})


def _make_init():
    nan = float("nan")
    return {
        "depth.is.on": "LuZ",
        "indice.water": 1.34,
        "rau.Fresnel": 0.043,
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 7.0, "LuZ": 7.0, "EuZ": 7.0},
        "delta.capteur.optics": {"Ed0": 0.0, "EdZ": DELTA_CAPTEUR_EDZ, "LuZ": DELTA_CAPTEUR_LUZ, "EuZ": 0.238},
        "sub.surface.removed.layer.optics": {"Ed0": 0.0, "EdZ": 0.3, "LuZ": 0.0, "EuZ": 0.0},
        "depth.interval.for.smoothing.optics": {"Ed0": 10.0, "EdZ": 3.0, "LuZ": 3.0, "EuZ": 3.0},
        "linear.fit.Rsquared.threshold.optics": {"Ed0": nan, "EdZ": 0.5, "LuZ": 0.5, "EuZ": 0.5},
        "linear.fit.max.delta.depth.optics": {"Ed0": nan, "EdZ": 3.0, "LuZ": 2.5, "EuZ": 2.5},
        "depth.discretization": [0, 0.01, 1, 0.02, 2, 0.05, 5, 0.1, 10, 0.2, 20, 0.5, 50, 1, 100, 2, 200, 5, 500],
    }


def test_process_cast_fits_every_present_instrument():
    ds = _make_dataset(include_edz=True)
    result = process_cast(ds, _make_init())

    assert set(result.instrument_fits) == {"EdZ", "LuZ"}
    assert "EuZ" not in result.instrument_fits


def test_process_cast_skips_absent_instrument():
    ds = _make_dataset(include_edz=False)
    result = process_cast(ds, _make_init())

    assert set(result.instrument_fits) == {"LuZ"}


def test_process_cast_computes_rrs_when_luz_present():
    ds = _make_dataset()
    result = process_cast(ds, _make_init())

    assert result.rrs_loess is not None
    assert result.rrs_linear is not None
    # gentler-attenuation wavelengths should give finite, positive Rrs
    assert np.all(result.rrs_linear.rrs_0p[2:] > 0)
    assert np.all(np.isfinite(result.rrs_linear.rrs_0p[2:]))


def test_process_cast_rrs_none_without_luz():
    ds = xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), np.full((300, 4), 100.0)),
            "EdZ": (("time", "wavelength"), np.full((300, 4), 50.0)),
            "Ed0_Roll": ("time", np.zeros(300)),
            "Ed0_Pitch": ("time", np.zeros(300)),
            "EdZ_Roll": ("time", np.zeros(300)),
            "EdZ_Pitch": ("time", np.zeros(300)),
            "LuZ_Depth": ("time", np.linspace(0.05, 6.0, 300)),
        },
        coords={"time": np.arange(300), "wavelength": np.array(WAVES)},
    )

    result = process_cast(ds, _make_init())

    assert result.rrs_loess is None
    assert result.rrs_linear is None
    assert set(result.instrument_fits) == {"EdZ"}


def test_process_cast_rrs_linear_matches_manual_compute_rrs():
    ds = _make_dataset()
    init = _make_init()
    result = process_cast(ds, init)

    from pycops.processing.rrs import compute_rrs

    expected = compute_rrs(
        result.instrument_fits["LuZ"].surface_linear.value_at_surface,
        result.ed0_fit.value_at_0,
        init["indice.water"],
        init["rau.Fresnel"],
    )
    np.testing.assert_allclose(result.rrs_linear.rrs_0p, expected.rrs_0p, equal_nan=True)
