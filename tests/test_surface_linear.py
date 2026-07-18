from __future__ import annotations

import numpy as np

from pycops.processing.surface_linear import fit_surface_linear


def test_recovers_known_exponential_decay():
    depth = np.linspace(0.2, 5.0, 400)  # time/depth-ordered downcast
    X0_true = 20.0
    K_true = 0.7
    aop = (X0_true * np.exp(-K_true * depth))[:, None]
    detection_limit = np.array([1e-4])

    fit = fit_surface_linear(depth, aop, detection_limit, r2_threshold=0.5, delta_depth=2.5)

    assert np.isfinite(fit.value_at_surface[0])
    np.testing.assert_allclose(fit.value_at_surface[0], X0_true, rtol=0.05)
    np.testing.assert_allclose(fit.k_surf[0], K_true, rtol=0.05)
    assert fit.r2[0] > 0.9


def test_returns_nan_when_all_below_detection_limit():
    depth = np.linspace(0.2, 5.0, 400)
    aop = (20.0 * np.exp(-0.7 * depth))[:, None]
    detection_limit = np.array([1e6])  # nothing clears this

    fit = fit_surface_linear(depth, aop, detection_limit)

    assert np.isnan(fit.value_at_surface[0])
    assert np.isnan(fit.k_surf[0])


def test_returns_nan_when_r2_below_threshold():
    rng = np.random.default_rng(1)
    depth = np.linspace(0.2, 5.0, 400)
    # pure noise, no real depth dependence -> should not fit well
    aop = np.abs(rng.normal(loc=1.0, scale=1.0, size=depth.shape))[:, None]
    detection_limit = np.array([1e-6])

    fit = fit_surface_linear(depth, aop, detection_limit, r2_threshold=0.95)

    assert np.isnan(fit.value_at_surface[0])


def test_multiple_wavelengths_independent():
    depth = np.linspace(0.2, 5.0, 400)
    aop = np.column_stack(
        [
            20.0 * np.exp(-0.7 * depth),
            10.0 * np.exp(-0.3 * depth),
        ]
    )
    detection_limit = np.array([1e-4, 1e-4])

    fit = fit_surface_linear(depth, aop, detection_limit, r2_threshold=0.5)

    np.testing.assert_allclose(fit.value_at_surface, [20.0, 10.0], rtol=0.05)
    np.testing.assert_allclose(fit.k_surf, [0.7, 0.3], rtol=0.05)
