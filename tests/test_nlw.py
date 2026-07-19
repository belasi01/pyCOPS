from __future__ import annotations

import numpy as np

from pycops.processing.nlw import compute_nlw, etirrwindow

WAVES = np.array([340, 380, 412, 443, 490, 510, 555, 620, 665, 683, 780], dtype=float)
# cross-checked by running copsutils.R's etirrwindow() directly (source it, load
# thuillier.completed.by.AM0AM1) with the same wavelengths and a 10 nm bandwidth.
EXPECTED_ETIRR_10NM = [
    94.464111, 111.257111, 175.620333, 191.746222, 195.564100,
    194.478125, 186.558778, 167.157444, 155.267000, 149.182900, 117.728000,
]


def test_etirrwindow_matches_real_r():
    result = etirrwindow(WAVES, 10)
    np.testing.assert_allclose(result, EXPECTED_ETIRR_10NM, atol=1e-4)


def test_etirrwindow_nan_outside_table_range():
    result = etirrwindow(np.array([100.0, 2000.0]), 10)
    assert np.all(np.isnan(result))


def test_etirrwindow_wider_window_smooths_more():
    narrow = etirrwindow(WAVES, 2)
    wide = etirrwindow(WAVES, 40)
    assert not np.allclose(narrow, wide)


def test_compute_nlw_matches_manual_formula():
    lw_0p = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0008, 0.0002])
    ed0_0p = np.full(WAVES.shape, 100.0)

    nlw = compute_nlw(lw_0p, ed0_0p, WAVES, 10)
    expected = lw_0p / ed0_0p * np.array(EXPECTED_ETIRR_10NM)
    np.testing.assert_allclose(nlw, expected, atol=1e-4)
