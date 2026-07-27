from __future__ import annotations

import numpy as np

from pycops.processing.par import percent_par_at_depth

WAVES = np.array([340, 380, 412, 443, 490, 510, 555, 620, 665, 683, 700], dtype=float)
K = np.array([2.0, 1.2, 0.8, 0.5, 0.3, 0.25, 0.15, 0.2, 0.35, 0.4, 0.45])
X0 = np.array([50, 70, 90, 100, 95, 90, 80, 40, 20, 15, 10], dtype=float)


def _profile(depths):
    return np.array([X0 * np.exp(-K * z) for z in depths])


def test_percent_par_matches_r_compute_par_fitted_r():
    """Validated against R's compute.PAR.fitted.R (sourced directly, same synthetic profile):
    R gives 26.76% -- within the small natural-cubic-spline-implementation discrepancy already
    documented elsewhere in this codebase (aop_cleaning.py, etc.), not an exact-match algorithm."""
    depth_grid = np.array([0.0, 5.0])
    pct = percent_par_at_depth(WAVES, _profile(depth_grid), depth_grid, 5.0)
    np.testing.assert_allclose(pct, 26.76221, rtol=0.02)


def test_percent_par_decreases_with_depth():
    depth_grid = np.linspace(0.0, 10.0, 50)
    profile = _profile(depth_grid)
    pct_shallow = percent_par_at_depth(WAVES, profile, depth_grid, 2.0)
    pct_deep = percent_par_at_depth(WAVES, profile, depth_grid, 8.0)
    assert 0 < pct_deep < pct_shallow < 100


def test_percent_par_no_decay_is_about_100_percent():
    depth_grid = np.linspace(0.0, 10.0, 20)
    flat_profile = np.tile(X0, (len(depth_grid), 1))
    pct = percent_par_at_depth(WAVES, flat_profile, depth_grid, 5.0)
    assert abs(pct - 100.0) < 1.0


def test_percent_par_none_when_surface_par_is_zero():
    depth_grid = np.array([0.0, 5.0])
    zero_profile = np.zeros((2, len(WAVES)))
    assert percent_par_at_depth(WAVES, zero_profile, depth_grid, 5.0) is None
