from __future__ import annotations

import numpy as np

from pycops.processing.ed0 import fit_ed0


def test_correction_compensates_cloud_dip():
    depth = np.linspace(0, 5, 200)
    waves = np.array([550.0])
    baseline = 100.0
    ed0_all = np.full((200, 1), baseline)
    dip_idx = 100
    ed0_all[dip_idx, 0] = baseline * 0.5  # a passing cloud briefly halves Ed0

    # QC'd subset excludes the dip (as tilt/depth QC would tend to smooth over
    # a brief instrument-level fluctuation less than removing it outright, but
    # for this test we simulate the fit being driven by the stable baseline).
    kept = np.ones(200, dtype=bool)
    kept[dip_idx] = False
    depth_grid = np.linspace(0, 5, 30)

    fit = fit_ed0(waves, depth[kept], ed0_all[kept], ed0_all, span=5.0, depth_grid=depth_grid, idx_depth_0=0)

    np.testing.assert_allclose(fit.value_at_0, [baseline], rtol=0.05)
    assert fit.correction[dip_idx, 0] > 1.5  # correction boosts the dipped scan back up
    np.testing.assert_allclose(fit.correction[0, 0], 1.0, rtol=0.05)


def test_correction_shape_matches_full_raw_matrix():
    depth = np.linspace(0, 5, 50)
    waves = np.array([440.0, 550.0])
    ed0_all = np.full((50, 2), 100.0)
    depth_grid = np.linspace(0, 5, 10)

    fit = fit_ed0(waves, depth, ed0_all, ed0_all, span=5.0, depth_grid=depth_grid)

    assert fit.correction.shape == ed0_all.shape
