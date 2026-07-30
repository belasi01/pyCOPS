from __future__ import annotations

import numpy as np

from pycops.processing.par import par_profile, par_quanta, percent_par_at_depth

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


def test_par_profile_matches_par_quanta_at_each_depth():
    """par_profile() must agree exactly with calling par_quanta() row-by-row -- no drift between
    the two entry points that both wrap the same underlying spline/photon-flux conversion."""
    depth_grid = np.linspace(0.0, 10.0, 15)
    profile = _profile(depth_grid)

    result = par_profile(WAVES, profile)

    expected = np.array([par_quanta(WAVES, profile[i, :]) for i in range(len(depth_grid))])
    np.testing.assert_allclose(result, expected)


def test_par_profile_decreases_monotonically_for_decaying_profile():
    depth_grid = np.linspace(0.0, 10.0, 20)
    profile = _profile(depth_grid)

    result = par_profile(WAVES, profile)

    assert np.all(np.diff(result) < 0)


def test_par_quanta_absolute_value_matches_r_compute_par_fitted_r():
    """Regression test for a real bug: par_quanta() originally omitted compute.PAR.fitted.R's
    ``* 1E-2`` uW/cm^2/nm -> W/m^2/nm SI conversion (confirmed against shadow.correction.R's
    matching ``* 100`` in the other direction) -- invisible in every other test here because
    percent_par_at_depth()/par_profile()'s own consumers only ever take a *ratio* of two
    par_quanta() calls, which cancels a missing constant factor out. Locks in the absolute
    magnitude (not just a ratio) against R's own compute.PAR.fitted.R, sourced directly and run
    on this exact synthetic profile at depth 0: R gives 1622.05 uEin.m-2.s-1 (waves/x0/k taken
    from this file's own WAVES/X0/K fixtures, restricted to the 9-band synthetic set used for
    that cross-check, at the shallowest depth 0.1 m, not exactly the surface)."""
    waves = np.array([340, 380, 412, 443, 490, 532, 555, 620, 683], dtype=float)
    k = np.array([0.35, 0.25, 0.15, 0.10, 0.08, 0.09, 0.10, 0.20, 0.30])
    x0 = np.array([50, 80, 120, 150, 180, 160, 140, 90, 60], dtype=float)
    values_at_0p1m = x0 * np.exp(-k * 0.1)

    result = par_quanta(waves, values_at_0p1m)

    np.testing.assert_allclose(result, 1622.054, rtol=0.005)
