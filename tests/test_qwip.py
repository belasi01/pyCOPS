from __future__ import annotations

import numpy as np

from pycops.processing.qwip import compute_qwip

WAVES = np.array([340, 380, 412, 443, 490, 510, 555, 620, 665, 683, 780], dtype=float)
RRS1 = np.array([0.0003, 0.0006, 0.0012, 0.0022, 0.0035, 0.0038, 0.0032, 0.0012, 0.0005, 0.0004, 0.00005])
RRS2 = np.array([0.0001, 0.0002, 0.0004, 0.0008, 0.0015, 0.0016, 0.0011, 0.0004, 0.00015, 0.0001, 0.00002])


def test_compute_qwip_matches_real_r_case1():
    # cross-checked by running QWIP.R's scoring formula directly with PLOT=FALSE
    result = compute_qwip(WAVES, RRS1)
    np.testing.assert_allclose(result.avw, 516.5148, atol=1e-3)
    np.testing.assert_allclose(result.ndi, -0.7525474, atol=1e-6)
    np.testing.assert_allclose(result.predicted_ndi, -0.6273498, atol=1e-6)
    np.testing.assert_allclose(result.score, -0.1251976, atol=1e-6)
    assert result.passed is False
    assert result.water_class == "Blue"
    assert result.fu == 6


def test_compute_qwip_matches_real_r_case2():
    result = compute_qwip(WAVES, RRS2)
    np.testing.assert_allclose(result.avw, 513.0983, atol=1e-3)
    np.testing.assert_allclose(result.ndi, -0.8202727, atol=1e-6)
    np.testing.assert_allclose(result.predicted_ndi, -0.6658142, atol=1e-6)
    np.testing.assert_allclose(result.score, -0.1544584, atol=1e-6)
    assert result.passed is False
    assert result.water_class == "Blue"


def test_compute_qwip_ignores_nan_wavelengths():
    rrs_with_nan = RRS1.copy()
    rrs_with_nan[-1] = np.nan

    result = compute_qwip(WAVES, rrs_with_nan)
    assert np.isfinite(result.avw)
    assert np.isfinite(result.score)


def test_compute_qwip_returns_none_with_fewer_than_two_finite_wavelengths():
    """Regression test: a cast whose linear fit only has one usable wavelength (e.g. every other
    band failed the R2/KS gate) used to crash CubicSpline construction instead of gracefully
    skipping QWIP for that cast -- found on a real CASCADE cast (Amundsen 2026, station CS1-9,
    CAST_002) whose linear Rrs had exactly one finite value, at 875 nm."""
    rrs_one_finite = np.full_like(RRS1, np.nan)
    rrs_one_finite[-1] = RRS1[-1]

    assert compute_qwip(WAVES, rrs_one_finite) is None

    rrs_zero_finite = np.full_like(RRS1, np.nan)
    assert compute_qwip(WAVES, rrs_zero_finite) is None
