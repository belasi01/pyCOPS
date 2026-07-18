from __future__ import annotations

import numpy as np

from pycops.processing.profile_fit import fit_profile_loess


def test_fit_recovers_exponential_decay():
    depth = np.linspace(0, 10, 300)
    waves = np.array([440.0, 550.0])
    K = np.array([0.5, 0.8])
    X0 = np.array([10.0, 5.0])
    aop = X0[None, :] * np.exp(-K[None, :] * depth[:, None])

    depth_grid = np.linspace(0, 10, 50)
    fit = fit_profile_loess(waves, depth, aop, span=2.0, depth_grid=depth_grid, idx_depth_0=0, depth_span=True)

    np.testing.assert_allclose(fit.value_at_0, X0, rtol=0.05)
    expected = X0[None, :] * np.exp(-K[None, :] * depth_grid[:, None])
    finite = np.isfinite(fit.fitted)
    np.testing.assert_allclose(fit.fitted[finite], expected[finite], rtol=0.1)


def test_fit_only_covers_from_idx_depth_0():
    depth = np.linspace(1, 10, 200)
    waves = np.array([440.0])
    aop = np.exp(-0.3 * depth)[:, None]
    depth_grid = np.linspace(0, 10, 20)
    idx0 = 5  # depth_grid[5] ~ 2.63, the first plausible in-range point

    fit = fit_profile_loess(waves, depth, aop, span=3.0, depth_grid=depth_grid, idx_depth_0=idx0)

    assert np.all(np.isnan(fit.fitted[:idx0, 0]))
    assert np.any(np.isfinite(fit.fitted[idx0:, 0]))


def test_fit_skips_wavelength_with_too_few_observations():
    depth = np.linspace(0, 10, 300)
    waves = np.array([440.0, 550.0])
    aop = np.column_stack(
        [
            np.exp(-0.3 * depth),
            np.full(300, np.nan),
        ]
    )
    aop[:2, 1] = 1.0  # only 2 finite points, below minimum_obs default of 3

    depth_grid = np.linspace(0, 10, 20)
    fit = fit_profile_loess(waves, depth, aop, span=2.0, depth_grid=depth_grid, minimum_obs=3)

    assert np.isnan(fit.value_at_0[1])
    assert np.all(np.isnan(fit.fitted[:, 1]))
    assert np.isfinite(fit.value_at_0[0])
