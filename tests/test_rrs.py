from __future__ import annotations

import numpy as np

from pycops.processing.rrs import compute_rrs


def test_compute_rrs_known_values():
    luz_0m = np.array([0.01, 0.02])
    ed0_0p = np.array([100.0, 200.0])
    indice_water = 1.34
    rau_fresnel = 0.043

    result = compute_rrs(luz_0m, ed0_0p, indice_water, rau_fresnel)

    expected_lw = luz_0m * (1 - rau_fresnel) / indice_water**2
    expected_rrs = expected_lw / ed0_0p
    np.testing.assert_allclose(result.lw_0p, expected_lw)
    np.testing.assert_allclose(result.rrs_0p, expected_rrs)


def test_compute_rrs_typical_oceanic_magnitude():
    # sanity: realistic LuZ(0-)/Ed0(0+) ratios should give Rrs in the typical
    # oceanic range (~1e-4 to 1e-1 sr^-1), not something wildly off-scale.
    result = compute_rrs(luz_0m=np.array([0.05]), ed0_0p=np.array([150.0]), indice_water=1.34, rau_fresnel=0.043)
    assert 1e-4 < result.rrs_0p[0] < 1e-1
