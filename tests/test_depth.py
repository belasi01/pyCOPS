from __future__ import annotations

import numpy as np
import pytest

from pycops.processing.depth import bin_by_depth, depth_grid, good_depth_mask, time_window_mask


def test_depth_grid_matches_expected_adaptive_points():
    grid = depth_grid([0, 0.1, 1, 0.5, 2])
    expected = np.concatenate([np.arange(0, 1.0, 0.1), [1.0, 1.5]])
    np.testing.assert_allclose(grid, expected)


def test_depth_grid_max_depth_truncates():
    grid = depth_grid([0, 0.1, 1, 0.5, 2], max_depth=1.0)
    assert grid[-1] == 1.0
    assert grid.max() <= 1.0


def test_depth_grid_real_discretization_is_monotonic_and_bounded():
    discretization = [0, 0.01, 1, 0.02, 2, 0.05, 5, 0.1, 10, 0.2, 20, 0.5, 50, 1, 100, 2, 200, 5, 500]
    grid = depth_grid(discretization, max_depth=15.0)
    assert grid[0] == 0.0
    assert np.all(np.diff(grid) > 0)
    assert grid.max() <= 15.0
    # fine resolution near the surface, coarser below 10 m
    assert np.isclose(grid[1] - grid[0], 0.01)
    assert grid[grid > 10][1] - grid[grid > 10][0] > 0.01


def test_good_depth_mask_flags_a_dropout():
    depth = np.linspace(0, 5, 200)
    depth[100] = -3.0
    mask = good_depth_mask(depth)
    assert mask[100] == False  # noqa: E712
    assert mask[90] == True  # noqa: E712
    assert mask[110] == True  # noqa: E712


def test_bin_by_depth_averages_into_bins():
    # 0.05 sits in bin 0.1's range, 0.3 and 0.4 in bin 0.2's, 1.05 in bin 1.0's
    # (kept well clear of the bin-edge midpoints to avoid float rounding flakiness).
    depth = np.array([0.05, 0.3, 0.4, 1.05])
    values = np.array([1.0, 2.0, 3.0, 10.0])
    grid = np.array([0.1, 0.2, 1.0])

    binned, counts = bin_by_depth(depth, values, grid)

    np.testing.assert_allclose(binned, [1.0, 2.5, 10.0])
    np.testing.assert_array_equal(counts, [1, 2, 1])


def test_bin_by_depth_respects_mask():
    depth = np.array([0.05, 0.3, 0.4, 1.05])
    values = np.array([1.0, 2.0, 3.0, 10.0])
    grid = np.array([0.1, 0.2, 1.0])
    mask = np.array([True, False, True, True])

    binned, counts = bin_by_depth(depth, values, grid, mask=mask)

    np.testing.assert_allclose(binned, [1.0, 3.0, 10.0])
    np.testing.assert_array_equal(counts, [1, 1, 1])


def test_bin_by_depth_2d_values_per_channel():
    depth = np.array([0.05, 0.3, 0.4, 1.05])
    values = np.column_stack([[1.0, 2.0, 3.0, 10.0], [10.0, 20.0, 30.0, 100.0]])
    grid = np.array([0.1, 0.2, 1.0])

    binned, counts = bin_by_depth(depth, values, grid)

    np.testing.assert_allclose(binned[:, 0], [1.0, 2.5, 10.0])
    np.testing.assert_allclose(binned[:, 1], [10.0, 25.0, 100.0])
    np.testing.assert_array_equal(counts, [1, 2, 1])


def test_bin_by_depth_empty_bin_is_nan():
    depth = np.array([0.05, 1.05])
    values = np.array([1.0, 10.0])
    grid = np.array([0.1, 0.2, 1.0])

    binned, counts = bin_by_depth(depth, values, grid)

    assert np.isnan(binned[1])
    assert counts[1] == 0


def test_time_window_mask_flags_scans_outside_window():
    time = np.datetime64("2020-01-01T00:00:00") + np.arange(10) * np.timedelta64(1, "s")
    mask = time_window_mask(time, (2.0, 6.0))
    np.testing.assert_array_equal(mask, [False, False, True, True, True, True, True, False, False, False])


def test_time_window_mask_no_restriction_when_window_covers_everything():
    time = np.datetime64("2020-01-01T00:00:00") + np.arange(5) * np.timedelta64(1, "s")
    mask = time_window_mask(time, (0.0, 100.0))
    assert mask.all()


def test_time_window_mask_degenerate_constant_timestamp_warns_and_keeps_all():
    time = np.full(5, np.datetime64("2020-01-01T00:00:00"))
    with pytest.warns(UserWarning, match="same timestamp"):
        mask = time_window_mask(time, (0.0, 1.0))
    assert mask.all()
