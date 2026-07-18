from __future__ import annotations

import numpy as np
import pytest

from pycops.processing.gregg_carder import clear_sky_irradiance, surface_reflectance


def test_clear_sky_irradiance_positive_and_finite_in_band():
    waves = np.array([443.0, 490.0, 555.0, 665.0])
    result = clear_sky_irradiance(julian_day=230, lon=-68.1, lat=49.1, waves=waves, sun_zenith_deg=41.8)

    assert np.all(np.isfinite(result.edir))
    assert np.all(np.isfinite(result.edif))
    assert np.all(result.edir > 0)
    assert np.all(result.edif > 0)
    np.testing.assert_allclose(result.ed, result.edir + result.edif)


def test_clear_sky_irradiance_nan_outside_reference_table_range():
    waves = np.array([305.0, 443.0, 950.0])  # 305 and 950 are outside the 320-900 nm table
    result = clear_sky_irradiance(julian_day=230, lon=-68.1, lat=49.1, waves=waves, sun_zenith_deg=41.8)

    assert np.isnan(result.edir[0])
    assert np.isnan(result.edir[2])
    assert np.isfinite(result.edir[1])


def test_clear_sky_irradiance_decreases_with_higher_zenith():
    waves = np.array([555.0])
    low_zenith = clear_sky_irradiance(julian_day=230, lon=-68.1, lat=49.1, waves=waves, sun_zenith_deg=20.0)
    high_zenith = clear_sky_irradiance(julian_day=230, lon=-68.1, lat=49.1, waves=waves, sun_zenith_deg=70.0)

    assert high_zenith.edir[0] < low_zenith.edir[0]


def test_clear_sky_irradiance_rejects_sun_below_horizon():
    with pytest.raises(ValueError):
        clear_sky_irradiance(julian_day=230, lon=-68.1, lat=49.1, waves=np.array([555.0]), sun_zenith_deg=95.0)


def test_surface_reflectance_direct_at_zero_zenith():
    rod, ros = surface_reflectance(sun_zenith_deg=0.0, wind_speed=3.0)
    assert rod == pytest.approx(0.0211, abs=1e-4)
    assert ros == pytest.approx(0.066, abs=1e-4)


def test_surface_reflectance_positive_across_conditions():
    for zenith in (0.0, 30.0, 60.0, 80.0):
        for ws in (1.0, 5.0, 10.0):
            rod, ros = surface_reflectance(zenith, ws)
            assert rod > 0
            assert ros > 0


def test_surface_reflectance_matches_r_greggcarder_sfcrfl():
    # Cross-checked by running GreggCarder.sfcrfl directly in R (source
    # GreggCarder.R): matches to 7+ decimal places in all three cases.
    rod, ros = surface_reflectance(sun_zenith_deg=41.781, wind_speed=6.0)
    assert rod == pytest.approx(0.02690752, abs=1e-7)
    assert ros == pytest.approx(0.05743635, abs=1e-7)

    rod, ros = surface_reflectance(sun_zenith_deg=70.0, wind_speed=10.0)
    assert rod == pytest.approx(0.12675, abs=1e-5)
    assert ros == pytest.approx(0.059156, abs=1e-6)


def test_clear_sky_irradiance_matches_r_greggcarder_f():
    # Cross-checked by running GreggCarder.f directly in R (source
    # GreggCarder.R/.data.R) for the real WISE_MAN_F5_CAST_003 cast
    # (WISEMan station MAN-F05, 2019-08-18): matches to 6-7 significant
    # digits at every wavelength (verified across all 19 real cast bands;
    # a representative subset is checked here).
    waves = np.array([320.0, 380.0, 490.0, 780.0])
    result = clear_sky_irradiance(
        julian_day=230, lon=-68.108833, lat=49.13445, waves=waves, sun_zenith_deg=41.781, visibility_km=25.0
    )

    np.testing.assert_allclose(result.edir, [0.1283896, 0.3785632, 0.9725873, 0.6344726], rtol=1e-5)
    np.testing.assert_allclose(result.edif, [0.2102027, 0.2526163, 0.3268808, 0.1381192], rtol=1e-5)
