from __future__ import annotations

import numpy as np

from pycops.processing.color import forel_ule_class

WAVES = np.array([340, 380, 412, 443, 490, 510, 555, 620, 665, 683, 780], dtype=float)

# cross-checked by running Rrs2FU.R directly (source it after loading CIE.RData)
# with the same wavelengths and spectra.
CASES = [
    ([0.0003, 0.0006, 0.0012, 0.0022, 0.0035, 0.0038, 0.0032, 0.0012, 0.0005, 0.0004, 0.00005], 0.2763556, 0.3818492, 6),
    ([0.0001, 0.0002, 0.0004, 0.0008, 0.0015, 0.0016, 0.0011, 0.0004, 0.00015, 0.0001, 0.00002], 0.2584537, 0.3794715, 6),
    ([0.0008, 0.0012, 0.0018, 0.0022, 0.0020, 0.0016, 0.0009, 0.0002, 0.00008, 0.00006, 0.00001], 0.1978401, 0.2383561, 3),
    ([0.00005, 0.0001, 0.0003, 0.0008, 0.0015, 0.0020, 0.0028, 0.0035, 0.0038, 0.0036, 0.0010], 0.43268, 0.421581, 16),
]


def test_forel_ule_class_matches_real_r():
    for rrs, expected_x, expected_y, expected_fu in CASES:
        result = forel_ule_class(WAVES, np.array(rrs))
        assert result.fu == expected_fu
        np.testing.assert_allclose(result.x, expected_x, atol=2e-4)
        np.testing.assert_allclose(result.y, expected_y, atol=2e-4)


def test_forel_ule_class_ignores_nan_wavelengths():
    rrs = np.array(CASES[0][0])
    rrs_with_nan = rrs.copy()
    rrs_with_nan[0] = np.nan

    result = forel_ule_class(WAVES, rrs_with_nan)
    assert result.fu == CASES[0][3]
