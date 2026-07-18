from __future__ import annotations

import numpy as np

from pycops.processing.attenuation import compute_K


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
