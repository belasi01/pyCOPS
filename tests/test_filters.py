from __future__ import annotations

import numpy as np

from pycops.processing.filters import median_filter


def test_median_filter_replaces_outlier():
    y = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0])
    f = median_filter(y, k=1, delta=0.5, replace=True)
    assert f[3] == 1.0
    np.testing.assert_allclose(f[[0, 1, 2, 4, 5, 6]], y[[0, 1, 2, 4, 5, 6]])


def test_median_filter_marks_nan_when_not_replacing():
    y = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0])
    f = median_filter(y, k=1, delta=0.5, replace=False)
    assert np.isnan(f[3])
    assert not np.isnan(f[2])


def test_median_filter_fill_false_blanks_edges():
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    f = median_filter(y, k=1, delta=0.0, fill=False, replace=True)
    assert np.isnan(f[0])
    assert np.isnan(f[-1])
    assert not np.isnan(f[2])


def test_median_filter_fill_true_keeps_edges():
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    f = median_filter(y, k=1, delta=0.0, fill=True, replace=True)
    assert f[0] == 1.0
    assert f[-1] == 1.0


def test_median_filter_too_short_returns_unchanged():
    y = np.array([1.0, 2.0, 3.0])
    f = median_filter(y, k=3, delta=0.0)
    np.testing.assert_allclose(f, y)


def test_median_filter_within_delta_untouched():
    y = np.array([1.0, 1.1, 0.9, 1.0, 1.0])
    f = median_filter(y, k=1, delta=1.0, replace=True)
    np.testing.assert_allclose(f, y)
