from __future__ import annotations

import numpy as np

from pycops.processing.ed0_0m import compute_ed0_0m, compute_ed0_subsurface

# cross-checked by running GreggCarder.sfcrfl() directly (source GreggCarder.R) and
# reproducing compute.aops.R's Ed0.0m formula by hand with the same inputs.
ED0_0P = np.array([50.0, 80.0, 100.0, 90.0, 60.0])
FED_DIR = np.array([0.9, 0.85, 0.8, 0.7, 0.5])


def test_compute_ed0_0m_matches_real_r_case1():
    edir = FED_DIR
    edif = 1.0 - FED_DIR
    result = compute_ed0_0m(ED0_0P, edir, edif, 41.8, 4.0)
    expected = [48.478255, 77.407140, 96.561341, 86.549556, 57.225503]
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_compute_ed0_0m_matches_real_r_case2():
    edir = FED_DIR
    edif = 1.0 - FED_DIR
    result = compute_ed0_0m(ED0_0P, edir, edif, 60.0, 8.0)
    expected = [46.554472, 74.535955, 93.230944, 84.017650, 56.158166]
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_compute_ed0_subsurface_r0m_ratio():
    edir = FED_DIR
    edif = 1.0 - FED_DIR
    euz_0 = np.array([1.0, 2.0, 3.0, 2.5, 1.0])
    euz_surf = np.array([1.1, 2.1, 3.1, 2.6, 1.1])

    result = compute_ed0_subsurface(ED0_0P, edir, edif, 41.8, 4.0, euz_0, euz_surf)

    np.testing.assert_allclose(result.fed_dir, FED_DIR)
    np.testing.assert_allclose(result.r0m_loess, euz_0 / result.ed0_0m)
    np.testing.assert_allclose(result.r0m_linear, euz_surf / result.ed0_0m)


def test_compute_ed0_0m_between_direct_and_diffuse_extremes():
    # Ed0.0m must lie between what pure-direct and pure-diffuse illumination would give
    ed0_0p = np.array([100.0])
    all_direct = compute_ed0_0m(ed0_0p, np.array([1.0]), np.array([1e-12]), 30.0, 5.0)
    all_diffuse = compute_ed0_0m(ed0_0p, np.array([1e-12]), np.array([1.0]), 30.0, 5.0)
    mixed = compute_ed0_0m(ed0_0p, np.array([0.5]), np.array([0.5]), 30.0, 5.0)

    lo, hi = sorted([all_direct[0], all_diffuse[0]])
    assert lo <= mixed[0] <= hi
