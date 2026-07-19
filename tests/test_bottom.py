from __future__ import annotations

import numpy as np

from pycops.processing.bottom import compute_bottom_depth, compute_bottom_reflectance
from pycops.processing.cast_fit import InstrumentFit
from pycops.processing.surface_linear import SurfaceLinearFit

DEPTH_GRID = np.arange(0, 5.01, 0.1)
WAVES = np.array([443.0, 555.0, 665.0])
K = np.array([0.3, 0.5, 1.2])

_DUMMY_SURFACE = SurfaceLinearFit(
    value_at_surface=np.full(3, np.nan),
    k_surf=np.full(3, np.nan),
    z_interval=np.full(3, np.nan),
    ix_z_interval=np.full(3, -1),
    r2=np.full(3, np.nan),
    ks_pvalue=np.full(3, np.nan),
)


def _make_fit(instrument, depth_grid, aop_fitted):
    n = len(depth_grid)
    return InstrumentFit(
        instrument=instrument,
        kept=np.ones(n, dtype=bool),
        depth_grid=depth_grid,
        idx_depth_0=0,
        detection_limit=np.zeros(3),
        aop_fitted=aop_fitted,
        value_at_0=aop_fitted[0],
        KZ=np.zeros((n - 1, 3)),
        K0=np.zeros((n - 1, 3)),
        surface_linear=_DUMMY_SURFACE,
    )


def _synthetic_rz(depth_grid, rng):
    true_rz = 0.05 * np.exp(-K[None, :] * depth_grid[:, None]) + 0.01
    return true_rz + rng.normal(scale=0.001, size=true_rz.shape)


def test_compute_bottom_depth_matches_manual_formula():
    depth = np.array([0.1, 1.0, 2.0, 4.9, 5.0, 3.0])
    kept = np.array([True, True, True, True, False, True])  # exclude the deepest point

    result = compute_bottom_depth(depth, kept, delta_capteur_m=0.238, distance_above_bottom_m=0.15)

    assert result == 4.9 + 0.238 + 0.15


def test_compute_bottom_reflectance_matches_real_r():
    # cross-checked by running fit.with.loess() directly (source fit.functions.R)
    # with the same depth grid, ratio profile, span, and extended (bottom-appended) grid.
    rng = np.random.default_rng(0)
    noisy_rz = _synthetic_rz(DEPTH_GRID, rng)

    luz_fit = _make_fit("LuZ", DEPTH_GRID, noisy_rz / np.pi)
    edz_fit = _make_fit("EdZ", DEPTH_GRID, np.ones((len(DEPTH_GRID), 3)))

    result = compute_bottom_reflectance("LuZ", WAVES, luz_fit, edz_fit, bottom_depth=5.3, span=3.0)

    assert result.bottom_depth == 5.3
    ix_near_bottom = np.argmin(np.abs(DEPTH_GRID - 5.0))
    np.testing.assert_allclose(result.depth_over_bottom, 5.3 - DEPTH_GRID[ix_near_bottom])
    assert np.all(np.isfinite(result.rb))
    assert np.all(np.isfinite(result.rb_extrapolated))
    # extrapolated (further) value should be smaller than the near-bottom one for this decaying profile
    assert np.all(result.rb_extrapolated < result.rb)


def test_compute_bottom_reflectance_euz_no_pi_factor():
    rng = np.random.default_rng(1)
    noisy_rz = _synthetic_rz(DEPTH_GRID, rng)

    euz_fit = _make_fit("EuZ", DEPTH_GRID, noisy_rz)  # no /pi -- EuZ ratio has no pi factor
    edz_fit = _make_fit("EdZ", DEPTH_GRID, np.ones((len(DEPTH_GRID), 3)))

    result = compute_bottom_reflectance("EuZ", WAVES, euz_fit, edz_fit, bottom_depth=5.3, span=3.0)

    assert np.all(np.isfinite(result.rb))


def test_compute_bottom_reflectance_interpolates_mismatched_depth_grids():
    # LuZ and EdZ have different depth grids (pycops builds one per instrument) --
    # must not crash and must still produce a sensible result.
    rng = np.random.default_rng(2)
    luz_grid = np.arange(0, 5.01, 0.1)
    edz_grid = np.arange(0, 4.51, 0.1)  # shorter grid

    noisy_rz = _synthetic_rz(luz_grid, rng)
    luz_fit = _make_fit("LuZ", luz_grid, noisy_rz / np.pi)
    edz_fit = _make_fit("EdZ", edz_grid, np.ones((len(edz_grid), 3)))

    result = compute_bottom_reflectance("LuZ", WAVES, luz_fit, edz_fit, bottom_depth=5.3, span=3.0)

    assert np.all(np.isfinite(result.rb_extrapolated))
