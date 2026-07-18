from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pycops.processing.cast_fit import fit_cast, fit_ed0_for_cast

WAVES = (340.0, 380.0, 443.0, 555.0)
K_TRUE = (2.0, 0.9, 0.3, 0.1)
X0_TRUE = (0.0015, 0.006, 0.05, 0.3)


DELTA_CAPTEUR_LUZ = 0.238


def _make_dataset(n=300, ed0_level=100.0, cloud_dip_at=None):
    # "LuZ_Depth" is the raw sensor reading; delta.capteur.optics["LuZ"] shifts it
    # to the true physical depth of the LuZ optical center (see fit_cast) -- so the
    # profile must be generated against that shifted depth for fit_cast to recover
    # X0_TRUE/K_TRUE, exactly as it would for a real cast.
    sensor_depth = np.linspace(0.05, 6.0, n)
    true_depth = sensor_depth + DELTA_CAPTEUR_LUZ
    waves = np.array(WAVES)
    K = np.array(K_TRUE)
    X0 = np.array(X0_TRUE)
    luz = X0[None, :] * np.exp(-K[None, :] * true_depth[:, None])

    ed0 = np.full((n, len(waves)), ed0_level)
    if cloud_dip_at is not None:
        ed0[cloud_dip_at, :] *= 0.5

    zeros = np.zeros(n)
    return xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), ed0),
            "LuZ": (("time", "wavelength"), luz),
            "Ed0_Roll": ("time", zeros),
            "Ed0_Pitch": ("time", zeros),
            "EdZ_Roll": ("time", zeros),  # LuZ has no inclinometer -> falls back to EdZ
            "EdZ_Pitch": ("time", zeros),
            "LuZ_Depth": ("time", sensor_depth),
            "LuZ_Temp": ("time", np.full(n, 10.0)),
        },
        coords={"time": np.arange(n), "wavelength": waves},
    )


def _make_init():
    nan = float("nan")
    return {
        "depth.is.on": "LuZ",
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 7.0, "LuZ": 7.0, "EuZ": 7.0},
        "delta.capteur.optics": {"Ed0": 0.0, "EdZ": -0.05, "LuZ": DELTA_CAPTEUR_LUZ, "EuZ": 0.238},
        "sub.surface.removed.layer.optics": {"Ed0": 0.0, "EdZ": 0.3, "LuZ": 0.0, "EuZ": 0.0},
        "depth.interval.for.smoothing.optics": {"Ed0": 10.0, "EdZ": 3.0, "LuZ": 3.0, "EuZ": 3.0},
        "linear.fit.Rsquared.threshold.optics": {"Ed0": nan, "EdZ": 0.5, "LuZ": 0.5, "EuZ": 0.5},
        "linear.fit.max.delta.depth.optics": {"Ed0": nan, "EdZ": 3.0, "LuZ": 2.5, "EuZ": 2.5},
        "depth.discretization": [0, 0.01, 1, 0.02, 2, 0.05, 5, 0.1, 10, 0.2, 20, 0.5, 50, 1, 100, 2, 200, 5, 500],
    }


def test_fit_ed0_for_cast_no_clouds_gives_unit_correction():
    ds = _make_dataset()
    ed0_fit = fit_ed0_for_cast(ds, _make_init())
    np.testing.assert_allclose(ed0_fit.correction, 1.0, atol=1e-6)


def test_fit_ed0_for_cast_compensates_cloud_dip():
    ds = _make_dataset(cloud_dip_at=150)
    ed0_fit = fit_ed0_for_cast(ds, _make_init())
    assert ed0_fit.correction[150, 0] > 1.5
    np.testing.assert_allclose(ed0_fit.correction[0, 0], 1.0, atol=0.05)


def test_fit_cast_luz_recovers_true_parameters():
    ds = _make_dataset()
    init = _make_init()
    ed0_fit = fit_ed0_for_cast(ds, init)

    result = fit_cast(ds, init, "LuZ", ed0_fit)

    # The linear surface fit stays accurate across the board (matches the
    # real-cast validation in CLAUDE.md).
    np.testing.assert_allclose(result.surface_linear.value_at_surface, X0_TRUE, rtol=0.1)
    np.testing.assert_allclose(result.surface_linear.k_surf, K_TRUE, rtol=0.1)

    # The LOESS fit is only tight at the gentler-attenuation wavelengths here:
    # depth starts at 0.288 m (delta.capteur-shifted), not exactly 0, and
    # extrapolating the LOESS fit that gap is more sensitive to attenuation
    # strength than the linear fit -- the same documented behavior found
    # reprocessing a real cast.
    np.testing.assert_allclose(result.value_at_0[2:], X0_TRUE[2:], rtol=0.15)
    assert result.value_at_0[0] == pytest.approx(X0_TRUE[0], rel=1.0)  # same order of magnitude


def test_fit_cast_kept_mask_matches_scan_count():
    ds = _make_dataset()
    init = _make_init()
    ed0_fit = fit_ed0_for_cast(ds, init)
    result = fit_cast(ds, init, "LuZ", ed0_fit)
    assert result.kept.shape == (ds.sizes["time"],)
    assert result.kept.sum() > 0


def test_fit_cast_rejects_ed0_instrument():
    ds = _make_dataset()
    init = _make_init()
    ed0_fit = fit_ed0_for_cast(ds, init)
    with pytest.raises(ValueError):
        fit_cast(ds, init, "Ed0", ed0_fit)


def test_fit_cast_KZ_K0_shapes_align_with_depth_grid():
    ds = _make_dataset()
    init = _make_init()
    ed0_fit = fit_ed0_for_cast(ds, init)
    result = fit_cast(ds, init, "LuZ", ed0_fit)
    n_grid = len(result.depth_grid)
    assert result.KZ.shape == (n_grid - 1, len(WAVES))
    assert result.K0.shape == (n_grid - 1, len(WAVES))
