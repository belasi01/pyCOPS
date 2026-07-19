from __future__ import annotations

import numpy as np

from pycops.processing.aop_cleaning import secondary_clean

DEPTH_GRID = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def _broader_qc_profile(n=20):
    # A near-linear decline (12 - 2*depth) crossing the detection limit
    # (5.0) around depth ~3.5 -- steeper than the smoother aop_fitted profile
    # below, so the secondary spline catches noise the primary fit's own
    # detection-limit masking missed.
    depth_qc = np.linspace(0.1, 5.9, n)
    aop_qc = 12.0 - 2.0 * depth_qc
    return depth_qc, aop_qc[:, None]


def test_secondary_clean_masks_points_primary_fit_missed():
    depth_qc, aop_qc = _broader_qc_profile()
    aop_fitted = np.array([[10.0], [9.0], [7.0], [6.0], [5.5], [5.2]])  # all > 5.0 limit
    value_at_0 = np.array([10.0])
    KZ = np.full((5, 1), 0.2)
    K0 = np.full((5, 1), 0.2)

    cleaned_fitted, cleaned_v0, cleaned_KZ, cleaned_K0 = secondary_clean(
        depth_qc,
        aop_qc,
        DEPTH_GRID,
        idx_depth_0=0,
        detection_limit=np.array([5.0]),
        depth_first_kept=0.5,
        aop_fitted=aop_fitted,
        value_at_0=value_at_0,
        KZ=KZ,
        K0=K0,
    )

    # Depths 4 and 5 (indices 4, 5): the broader QC'd data actually dips
    # below the detection limit there even though the primary fit (5.5, 5.2)
    # stayed just above it.
    assert np.isnan(cleaned_fitted[4, 0])
    assert np.isnan(cleaned_fitted[5, 0])
    np.testing.assert_allclose(cleaned_fitted[:4, 0], [10.0, 9.0, 7.0, 6.0])

    # KZ/K0 are aligned with depth_grid[1:] = [1, 2, 3, 4, 5]; mask offsets
    # by one relative to aop_fitted's own index.
    assert np.isnan(cleaned_KZ[3, 0])  # depth 4
    assert np.isnan(cleaned_KZ[4, 0])  # depth 5
    assert not np.isnan(cleaned_KZ[0, 0])
    assert np.isnan(cleaned_K0[3, 0])
    assert np.isnan(cleaned_K0[4, 0])

    # value_at_0 tracks aop_fitted at idx_depth_0, which wasn't masked.
    assert cleaned_v0[0] == 10.0


def test_secondary_clean_guard_excludes_shallow_points():
    # Even where the spline dips below the limit, points shallower than (or
    # at) depth_first_kept must not be masked (matches R's `Depth[1]` guard).
    depth_qc = np.linspace(0.01, 5.9, 20)
    aop_qc = (2.0 - 2.0 * depth_qc)[:, None]  # already below 5.0 limit everywhere
    aop_fitted = np.full((6, 1), 10.0)  # primary fit stays "clean" everywhere
    value_at_0 = np.array([10.0])
    KZ = np.full((5, 1), 0.2)
    K0 = np.full((5, 1), 0.2)

    cleaned_fitted, _, _, _ = secondary_clean(
        depth_qc,
        aop_qc,
        DEPTH_GRID,
        idx_depth_0=0,
        detection_limit=np.array([5.0]),
        depth_first_kept=10.0,  # guard excludes every grid point
        aop_fitted=aop_fitted,
        value_at_0=value_at_0,
        KZ=KZ,
        K0=K0,
    )
    assert not np.any(np.isnan(cleaned_fitted))


def test_secondary_clean_skips_all_nan_wavelength():
    depth_qc, aop_qc_ok = _broader_qc_profile()
    aop_qc = np.concatenate([aop_qc_ok, aop_qc_ok], axis=1)
    aop_fitted = np.full((6, 2), np.nan)
    aop_fitted[:, 0] = [10.0, 9.0, 7.0, 6.0, 5.5, 5.2]
    value_at_0 = np.array([10.0, np.nan])
    KZ = np.full((5, 2), 0.2)
    K0 = np.full((5, 2), 0.2)

    cleaned_fitted, cleaned_v0, cleaned_KZ, cleaned_K0 = secondary_clean(
        depth_qc,
        aop_qc,
        DEPTH_GRID,
        idx_depth_0=0,
        detection_limit=np.array([5.0, 5.0]),
        depth_first_kept=0.5,
        aop_fitted=aop_fitted,
        value_at_0=value_at_0,
        KZ=KZ,
        K0=K0,
    )
    # Wavelength 1 was already all-NaN and must be left untouched (no crash,
    # no attempted spline fit).
    assert np.all(np.isnan(cleaned_fitted[:, 1]))
    assert np.isnan(cleaned_v0[1])
    np.testing.assert_allclose(cleaned_KZ[:, 1], 0.2)  # KZ untouched, not masked
    np.testing.assert_allclose(cleaned_K0[:, 1], 0.2)
    # Wavelength 0 is still cleaned as in the first test.
    assert np.isnan(cleaned_fitted[4, 0])


def test_secondary_clean_falls_back_to_primary_when_qc_data_too_sparse():
    depth_qc = np.array([0.5, 1.0, 2.0])  # far fewer than the minimum for a spline
    aop_qc = np.array([[10.0], [8.0], [6.0]])
    aop_fitted = np.array([[10.0], [9.0], [7.0], [6.0], [4.0], [2.0]])
    value_at_0 = np.array([10.0])
    KZ = np.full((5, 1), 0.2)
    K0 = np.full((5, 1), 0.2)

    cleaned_fitted, _, _, _ = secondary_clean(
        depth_qc,
        aop_qc,
        DEPTH_GRID,
        idx_depth_0=0,
        detection_limit=np.array([5.0]),
        depth_first_kept=0.5,
        aop_fitted=aop_fitted,
        value_at_0=value_at_0,
        KZ=KZ,
        K0=K0,
    )
    # No spline fit possible -> only the primary detection-limit check applies.
    assert np.isnan(cleaned_fitted[4, 0])
    assert np.isnan(cleaned_fitted[5, 0])
    np.testing.assert_allclose(cleaned_fitted[:4, 0], [10.0, 9.0, 7.0, 6.0])
