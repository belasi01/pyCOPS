from __future__ import annotations

import numpy as np

from pycops.processing.loess import loess_1d


def test_loess_recovers_exact_line_noise_free():
    x = np.linspace(0, 10, 50)
    y = 2.0 * x + 3.0
    xout = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    fitted = loess_1d(x, y, xout, span=0.5)
    np.testing.assert_allclose(fitted, 2.0 * xout + 3.0, atol=1e-8)


def test_loess_recovers_exact_quadratic_noise_free():
    x = np.linspace(-5, 5, 60)
    y = 1.5 * x**2 - 2.0 * x + 0.5
    xout = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    fitted = loess_1d(x, y, xout, span=0.6, degree=2)
    expected = 1.5 * xout**2 - 2.0 * xout + 0.5
    np.testing.assert_allclose(fitted, expected, atol=1e-6)


def test_loess_smooths_noise_towards_mean():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    y = np.full_like(x, 5.0) + rng.normal(scale=0.5, size=x.size)
    fitted = loess_1d(x, y, xout=np.array([5.0]), span=0.8)
    assert abs(fitted[0] - 5.0) < 0.2


def test_loess_extrapolates_beyond_data_range():
    x = np.linspace(1, 10, 50)
    y = 2.0 * x + 1.0
    fitted = loess_1d(x, y, xout=np.array([0.0]), span=0.3)
    np.testing.assert_allclose(fitted, [1.0], atol=0.5)


def test_loess_smaller_span_follows_data_more_closely():
    x = np.linspace(0, 10, 100)
    y = np.where(x < 5, 0.0, 10.0)  # step function
    xout = np.array([5.5])  # just past the step, where narrow vs wide spans diverge
    tight = loess_1d(x, y, xout, span=0.1)
    wide = loess_1d(x, y, xout, span=0.9)
    assert abs(tight[0] - 10.0) < abs(wide[0] - 10.0)
