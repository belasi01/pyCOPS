from __future__ import annotations

import numpy as np
import xarray as xr

from pycops.processing.tilt import add_tilt, compute_tilt, tilt_mask


def test_compute_tilt_zero_when_level():
    tilt = compute_tilt(np.array([0.0]), np.array([0.0]))
    np.testing.assert_allclose(tilt, [0.0])


def test_compute_tilt_known_angle():
    tilt = compute_tilt(np.array([45.0]), np.array([0.0]))
    np.testing.assert_allclose(tilt, [45.0], atol=1e-9)


def test_compute_tilt_symmetric_in_roll_and_pitch():
    tilt_roll = compute_tilt(np.array([10.0]), np.array([0.0]))
    tilt_pitch = compute_tilt(np.array([0.0]), np.array([10.0]))
    np.testing.assert_allclose(tilt_roll, tilt_pitch)


def _sample_dataset():
    return xr.Dataset(
        {
            "Ed0_Roll": ("time", np.array([0.0, 45.0, 0.0])),
            "Ed0_Pitch": ("time", np.array([0.0, 0.0, 45.0])),
        },
        coords={"time": np.arange(3)},
    )


def test_add_tilt_matches_compute_tilt():
    ds = _sample_dataset()
    out = add_tilt(ds, "Ed0")
    expected = compute_tilt(ds["Ed0_Roll"].values, ds["Ed0_Pitch"].values)
    np.testing.assert_allclose(out["Ed0_Tilt"].values, expected)


def test_add_tilt_overwrites_existing_column():
    ds = _sample_dataset()
    ds = ds.assign({"Ed0_Tilt": ("time", np.array([999.0, 999.0, 999.0]))})
    out = add_tilt(ds, "Ed0")
    assert out["Ed0_Tilt"].values[0] != 999.0


def test_tilt_mask_filters_above_threshold():
    ds = _sample_dataset()
    mask = tilt_mask(ds, "Ed0", tiltmax=10.0)
    np.testing.assert_array_equal(mask.values, [True, False, False])


def test_luz_falls_back_to_edz_roll_pitch():
    # LuZ shares a frame with EdZ and has no inclinometer of its own in the
    # raw files (confirmed on real WISEMan/AlgaeWISE casts).
    ds = xr.Dataset(
        {
            "EdZ_Roll": ("time", np.array([45.0])),
            "EdZ_Pitch": ("time", np.array([0.0])),
        },
        coords={"time": np.arange(1)},
    )
    out = add_tilt(ds, "LuZ")
    np.testing.assert_allclose(out["LuZ_Tilt"].values, [45.0], atol=1e-9)


def test_luz_falls_back_to_euz_when_edz_also_missing():
    ds = xr.Dataset(
        {
            "EuZ_Roll": ("time", np.array([45.0])),
            "EuZ_Pitch": ("time", np.array([0.0])),
        },
        coords={"time": np.arange(1)},
    )
    out = add_tilt(ds, "LuZ")
    np.testing.assert_allclose(out["LuZ_Tilt"].values, [45.0], atol=1e-9)


def test_missing_roll_pitch_raises_clear_error():
    ds = xr.Dataset({}, coords={"time": np.arange(1)})
    try:
        add_tilt(ds, "LuZ")
        raise AssertionError("expected KeyError")
    except KeyError as e:
        assert "LuZ" in str(e)
