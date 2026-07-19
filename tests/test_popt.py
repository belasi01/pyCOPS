from __future__ import annotations

import numpy as np

from pycops.processing.popt import chlorophyll_absorption

RADIUS = 0.035
WAVES_A = np.array([320, 340, 380, 443, 555, 665, 780, 875], dtype=float)
WAVES_B = np.array([305, 320, 412, 490, 510, 620, 670, 700, 705, 750, 865, 900], dtype=float)


def test_chlorophyll_absorption_matches_real_r_low_chl():
    # cross-checked by running popt.R directly (source popt.R, reproduce
    # shadow.correction.R's clamped chlTMP + 3-band wavelength handling)
    expected_aR = [0.000736, 0.000736, 0.000529, 0.000449, 0.001977, 0.013849, 0.090125, 0.194249]
    a = chlorophyll_absorption(WAVES_A, 0.0)
    np.testing.assert_allclose(a * RADIUS, expected_aR, atol=1e-5)


def test_chlorophyll_absorption_chl_below_min_clamps_same_as_chl_zero():
    # chl=0 and chl=0.03 both clamp to chlTMP=0.03001 in the R source
    a0 = chlorophyll_absorption(WAVES_A, 0.0)
    a_small = chlorophyll_absorption(WAVES_A, 0.03)
    np.testing.assert_allclose(a0, a_small)


def test_chlorophyll_absorption_matches_real_r_mid_chl():
    expected_aR = [0.008716, 0.008716, 0.006468, 0.005565, 0.003519, 0.016063, 0.090125, 0.194249]
    a = chlorophyll_absorption(WAVES_A, 2.5)
    np.testing.assert_allclose(a * RADIUS, expected_aR, atol=1e-5)


def test_chlorophyll_absorption_chl_above_max_clamps_same_as_ten():
    # chl=10 and chl=15 both clamp to chlTMP=9.99999 in the R source
    a10 = chlorophyll_absorption(WAVES_A, 10.0)
    a15 = chlorophyll_absorption(WAVES_A, 15.0)
    np.testing.assert_allclose(a10, a15)

    expected_aR = [0.024781, 0.024781, 0.016396, 0.013945, 0.006155, 0.020345, 0.090125, 0.194249]
    np.testing.assert_allclose(a10 * RADIUS, expected_aR, atol=1e-5)


def test_chlorophyll_absorption_matches_real_r_second_wave_set():
    for chl, expected_aR in (
        (1.5, [0.006007, 0.006007, 0.004346, 0.002947, 0.002999, 0.009948, 0.015812, 0.020686, 0.024640, 0.070700, 0.194249, 0.194249]),
        (5.0, [0.014624, 0.014624, 0.009476, 0.006234, 0.005634, 0.011425, 0.018311, 0.021619, 0.024640, 0.070700, 0.194249, 0.194249]),
    ):
        a = chlorophyll_absorption(WAVES_B, chl)
        np.testing.assert_allclose(a * RADIUS, expected_aR, atol=1e-5)


def test_chlorophyll_absorption_uv_band_reuses_single_value():
    # every wavelength below 350 nm must get the identical value (a single
    # popt.f.a(350.001, chlTMP) call reused for all of them, per the R source)
    a = chlorophyll_absorption(WAVES_B, 5.0)
    uv = WAVES_B < 350.0
    assert uv.sum() >= 2
    assert np.all(a[uv] == a[uv][0])


def test_chlorophyll_absorption_ir_band_independent_of_chl():
    # above 700 nm the model uses pure-water absorption only -- no chlorophyll term
    a_low = chlorophyll_absorption(WAVES_B, 0.5)
    a_high = chlorophyll_absorption(WAVES_B, 8.0)
    ir = WAVES_B > 700.0
    np.testing.assert_allclose(a_low[ir], a_high[ir])
