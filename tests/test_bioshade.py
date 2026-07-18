from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pycops.processing.bioshade import process_bioshade

WAVES = (340.0, 443.0, 490.0, 665.0)
ED0_TOT_TRUE = (80.0, 120.0, 150.0, 60.0)
DIFFUSE_FRACTION_TRUE = 0.3


def _make_dataset(n=300, occlusion_index=150, position_period_s=60.0):
    time_s = np.linspace(0, n - 1, n)  # 1 Hz
    times = np.datetime64("2024-06-01T16:00:00") + (time_s * 1e9).astype("timedelta64[ns]")

    # continuous rotation of the shading band, wrapping every position_period_s
    position = (time_s % position_period_s) / position_period_s * 26000.0

    ed0_tot = np.array(ED0_TOT_TRUE)
    ed0 = np.tile(ed0_tot, (n, 1))
    ed0[occlusion_index, :] = ed0_tot * DIFFUSE_FRACTION_TRUE  # the sun-occlusion scan

    zeros = np.zeros(n)
    return xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), ed0),
            "Ed0_Roll": ("time", zeros),
            "Ed0_Pitch": ("time", zeros),
            "BioShadePosition": ("time", position),
        },
        coords={"time": times, "wavelength": np.array(WAVES)},
    )


def _make_init():
    return {
        "tiltmax.optics": {"Ed0": 10.0},
        "time.window": [0.0, 10000.0],
    }


def test_process_bioshade_recovers_true_diffuse_fraction():
    ds = _make_dataset()
    result = process_bioshade(ds, _make_init())

    np.testing.assert_allclose(result.ed0_tot, ED0_TOT_TRUE, rtol=0.05)
    np.testing.assert_allclose(result.ed0_dif, np.array(ED0_TOT_TRUE) * DIFFUSE_FRACTION_TRUE, rtol=0.05)
    np.testing.assert_allclose(result.ed0_diffuse_fraction, DIFFUSE_FRACTION_TRUE, rtol=0.1)


def test_process_bioshade_shape_matches_waves():
    ds = _make_dataset()
    result = process_bioshade(ds, _make_init())

    assert result.waves.shape == (len(WAVES),)
    assert result.ed0_tot.shape == (len(WAVES),)
    assert result.ed0_dif.shape == (len(WAVES),)
    assert result.ed0_diffuse_fraction.shape == (len(WAVES),)


def test_process_bioshade_missing_position_column_raises():
    ds = _make_dataset()
    ds = ds.drop_vars("BioShadePosition")
    with pytest.raises(KeyError):
        process_bioshade(ds, _make_init())


def test_process_bioshade_accepts_alternate_position_column_name():
    ds = _make_dataset()
    ds = ds.rename({"BioShadePosition": "BioShade_Position"})
    result = process_bioshade(ds, _make_init())
    np.testing.assert_allclose(result.ed0_diffuse_fraction, DIFFUSE_FRACTION_TRUE, rtol=0.1)


def test_process_bioshade_finds_occlusion_at_different_index():
    ds = _make_dataset(occlusion_index=75)
    result = process_bioshade(ds, _make_init())
    np.testing.assert_allclose(result.ed0_diffuse_fraction, DIFFUSE_FRACTION_TRUE, rtol=0.1)
