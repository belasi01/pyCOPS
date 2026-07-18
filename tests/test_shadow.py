from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pycops.processing.process_cast import process_cast
from pycops.processing.shadow import (
    kd_derived_absorption,
    resolve_absorption,
    shadow_correction,
    shadow_epsilon,
)

WAVES = (340.0, 380.0, 443.0, 555.0)
K_TRUE = (2.0, 0.9, 0.3, 0.1)
LUZ_X0_TRUE = (0.0015, 0.006, 0.05, 0.3)
EDZ_X0_TRUE = (50.0, 80.0, 150.0, 300.0)
DELTA_CAPTEUR_LUZ = 0.238
DELTA_CAPTEUR_EDZ = -0.05


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


def _cast_result():
    return process_cast(_make_dataset(), _make_init())


def test_shadow_epsilon_basic_properties():
    aR = np.array([0.0, 0.05, 0.2])
    result = shadow_epsilon("LuZ", aR, sun_zenith_deg=40.0, ratio_edsky_edsun=np.array([0.5, 0.5, 0.5]))

    assert result.eps_sun[0] == 0.0  # no absorption, no shading
    assert np.all(result.eps_sun[1:] > 0)
    assert np.all(result.eps_sky[1:] > 0)
    assert np.all((result.eps >= 0) & (result.eps < 1))


def test_shadow_epsilon_matches_r_shadow_epsilon_exactly():
    # Cross-checked by running shadow.epsilon() directly in R (source
    # shadow.data.R/shadow.epsilon.R) with the same inputs: matches to 7+
    # decimal places for both LuZ and EuZ.
    aR = np.array([0.0784, 0.0625, 0.15, 0.3])
    ratio = np.array([0.7441, 0.6513, 1.0, 0.5])

    luz = shadow_epsilon("LuZ", aR, sun_zenith_deg=41.781, ratio_edsky_edsun=ratio)
    np.testing.assert_allclose(luz.eps_sun, [0.2680882, 0.2202644, 0.4496061, 0.6970665], rtol=1e-6)
    np.testing.assert_allclose(luz.eps_sky, [0.3033165, 0.2503321, 0.4991757, 0.7491751], rtol=1e-6)
    np.testing.assert_allclose(luz.eps, [0.2831179, 0.2321236, 0.4743909, 0.714436], rtol=1e-6)

    euz = shadow_epsilon("EuZ", aR, sun_zenith_deg=41.781, ratio_edsky_edsun=ratio)
    np.testing.assert_allclose(euz.eps_sun, [0.3129732, 0.2586276, 0.5123734, 0.7622203], rtol=1e-6)
    np.testing.assert_allclose(euz.eps_sky, [0.1907764, 0.1552799, 0.3330232, 0.5551419], rtol=1e-6)
    np.testing.assert_allclose(euz.eps, [0.2608394, 0.2178656, 0.4226983, 0.6931942], rtol=1e-6)


def test_shadow_epsilon_invalid_instrument_raises():
    with pytest.raises(ValueError):
        shadow_epsilon("EdZ", np.array([0.1]), 40.0, np.array([0.5]))


def test_resolve_absorption_nan_chl_disables_correction():
    result = resolve_absorption("LuZ", float("nan"), _cast_result())
    assert result is None


def test_resolve_absorption_chl_zero_uses_absorption_file():
    waves = np.array([340.0, 380.0, 443.0, 555.0])
    values = np.array([5.0, 3.0, 1.0, 0.5])
    result = resolve_absorption("LuZ", 0.0, _cast_result(), absorption_waves=waves, absorption_values=values)

    assert result.source == "file"
    np.testing.assert_allclose(result.waves, waves)
    np.testing.assert_allclose(result.values, values)


def test_resolve_absorption_chl_zero_requires_absorption_data():
    with pytest.raises(ValueError):
        resolve_absorption("LuZ", 0.0, _cast_result())


def test_resolve_absorption_chl_999_uses_kd():
    cast_result = _cast_result()
    result = resolve_absorption("LuZ", 999.0, cast_result)

    assert result.source == "kd"
    np.testing.assert_array_equal(result.waves, cast_result.waves)
    assert result.values.shape == cast_result.waves.shape


def test_resolve_absorption_chl_positive_not_implemented():
    with pytest.raises(NotImplementedError):
        resolve_absorption("LuZ", 2.5, _cast_result())


def test_kd_derived_absorption_finite_where_fits_succeed():
    cast_result = _cast_result()
    values = kd_derived_absorption("LuZ", cast_result)
    # gentler-attenuation wavelengths (index 2, 3) should have a usable fit
    assert np.all(np.isfinite(values[2:]))


def test_shadow_correction_end_to_end_sane_values():
    cast_result = _cast_result()
    absorption = resolve_absorption("LuZ", 999.0, cast_result)

    result = shadow_correction(
        instrument="LuZ",
        waves=cast_result.waves,
        absorption=absorption,
        radius_m=0.035,
        sun_zenith_deg=41.8,
        julian_day=230,
        lon=-68.108833,
        lat=49.13445,
        ed0_0p=cast_result.ed0_fit.value_at_0,
    )

    finite = np.isfinite(result.correction)
    assert finite.any()
    assert np.all((result.correction[finite] > 0) & (result.correction[finite] <= 1))
    assert np.all(result.edif[np.isfinite(result.edif)] > 0)
    assert np.all(result.edir[np.isfinite(result.edir)] > 0)
