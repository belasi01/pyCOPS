from __future__ import annotations

import numpy as np

from pycops.processing.attenuation import compute_K, kd_at_light_fraction


def test_compute_K_recovers_constant_attenuation():
    depth_grid = np.linspace(0, 10, 50)
    K_true = np.array([0.3, 0.6])
    X0 = np.array([10.0, 2.0])
    fitted = X0[None, :] * np.exp(-K_true[None, :] * depth_grid[:, None])
    value_at_0 = fitted[0]

    KZ, K0 = compute_K(depth_grid, idx_depth_0=0, value_at_0=value_at_0, fitted=fitted)

    np.testing.assert_allclose(KZ, np.broadcast_to(K_true, KZ.shape), rtol=1e-6)
    np.testing.assert_allclose(K0, np.broadcast_to(K_true, K0.shape), rtol=1e-6)


def test_compute_K_shapes():
    depth_grid = np.linspace(0, 5, 20)
    fitted = np.exp(-0.4 * depth_grid)[:, None]
    KZ, K0 = compute_K(depth_grid, idx_depth_0=0, value_at_0=fitted[0], fitted=fitted)
    assert KZ.shape == (19, 1)
    assert K0.shape == (19, 1)


def test_compute_K0_matches_manual_formula_for_non_exponential_profile():
    depth_grid = np.array([0.0, 1.0, 2.0, 3.0])
    fitted = np.array([[10.0], [5.0], [3.0], [1.0]])
    value_at_0 = fitted[0]

    _, K0 = compute_K(depth_grid, idx_depth_0=0, value_at_0=value_at_0, fitted=fitted)

    expected = (np.log(value_at_0[0]) - np.log(fitted[1:, 0])) / (depth_grid[1:] - depth_grid[0])
    np.testing.assert_allclose(K0[:, 0], expected)


def test_kd_at_light_fraction_recovers_constant_attenuation():
    depth_grid = np.linspace(0, 20, 400)
    K_true = np.array([0.3, 0.9, 2.0])
    ed0_subsurface = np.array([100.0, 50.0, 10.0])
    fitted = ed0_subsurface[None, :] * np.exp(-K_true[None, :] * depth_grid[:, None])

    for fraction in (0.01, 0.1, 1 / np.e):
        kd = kd_at_light_fraction(fitted, depth_grid, ed0_subsurface, fraction)
        np.testing.assert_allclose(kd, K_true, rtol=1e-3)


def test_kd_at_light_fraction_nan_when_fraction_never_reached():
    # K=0.05 over a 5 m profile only attenuates to exp(-0.25) ~ 0.78 -- the 1% level (0.01) is
    # never observed, so this must return NaN rather than clip to the deepest measured point.
    depth_grid = np.linspace(0, 5, 50)
    ed0_subsurface = np.array([100.0])
    fitted = ed0_subsurface[None, :] * np.exp(-0.05 * depth_grid[:, None])

    kd = kd_at_light_fraction(fitted, depth_grid, ed0_subsurface, 0.01)

    assert np.isnan(kd[0])


def test_kd_at_light_fraction_nan_propagates_from_missing_fitted_values():
    depth_grid = np.linspace(0, 10, 50)
    ed0_subsurface = np.array([100.0, 100.0])
    fitted = ed0_subsurface[None, :] * np.exp(-np.array([0.5, 0.5])[None, :] * depth_grid[:, None])
    fitted[:, 1] = np.nan  # second wavelength entirely missing (e.g. below detection limit)

    kd = kd_at_light_fraction(fitted, depth_grid, ed0_subsurface, 0.1)

    assert np.isfinite(kd[0])
    assert np.isnan(kd[1])
