from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from pycops.processing.bioshade import BioShadeResult
from pycops.processing.position import PositionOverride
from pycops.processing.process_cast import process_cast
from pycops.processing.rrs import compute_rrs

WAVES = (340.0, 380.0, 443.0, 555.0)
K_TRUE = (2.0, 0.9, 0.3, 0.1)
LUZ_X0_TRUE = (0.0015, 0.006, 0.05, 0.3)
EDZ_X0_TRUE = (50.0, 80.0, 150.0, 300.0)
DELTA_CAPTEUR_LUZ = 0.238
DELTA_CAPTEUR_EDZ = -0.05


def _make_dataset(n=300, ed0_level=100.0, include_edz=True):
    sensor_depth = np.linspace(0.05, 6.0, n)
    waves = np.array(WAVES)
    K = np.array(K_TRUE)

    luz_true_depth = sensor_depth + DELTA_CAPTEUR_LUZ
    luz = np.array(LUZ_X0_TRUE)[None, :] * np.exp(-K[None, :] * luz_true_depth[:, None])

    ed0 = np.full((n, len(waves)), ed0_level)
    zeros = np.zeros(n)

    data_vars = {
        "Ed0": (("time", "wavelength"), ed0),
        "LuZ": (("time", "wavelength"), luz),
        "Ed0_Roll": ("time", zeros),
        "Ed0_Pitch": ("time", zeros),
        "EdZ_Roll": ("time", zeros),  # LuZ has no inclinometer -> falls back to EdZ
        "EdZ_Pitch": ("time", zeros),
        "LuZ_Depth": ("time", sensor_depth),
        "LuZ_Temp": ("time", np.full(n, 10.0)),
    }
    if include_edz:
        edz_true_depth = sensor_depth + DELTA_CAPTEUR_EDZ
        data_vars["EdZ"] = (
            ("time", "wavelength"),
            np.array(EDZ_X0_TRUE)[None, :] * np.exp(-K[None, :] * edz_true_depth[:, None]),
        )

    return xr.Dataset(data_vars, coords={"time": np.arange(n), "wavelength": waves})


def _make_init():
    nan = float("nan")
    return {
        "depth.is.on": "LuZ",
        "indice.water": 1.34,
        "rau.Fresnel": 0.043,
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 7.0, "LuZ": 7.0, "EuZ": 7.0},
        "radius.instrument.optics": {"Ed0": 0.035, "EdZ": 0.035, "LuZ": 0.035, "EuZ": 0.035},
        "delta.capteur.optics": {"Ed0": 0.0, "EdZ": DELTA_CAPTEUR_EDZ, "LuZ": DELTA_CAPTEUR_LUZ, "EuZ": 0.238},
        "sub.surface.removed.layer.optics": {"Ed0": 0.0, "EdZ": 0.3, "LuZ": 0.0, "EuZ": 0.0},
        "depth.interval.for.smoothing.optics": {"Ed0": 10.0, "EdZ": 3.0, "LuZ": 3.0, "EuZ": 3.0},
        "linear.fit.Rsquared.threshold.optics": {"Ed0": nan, "EdZ": 0.5, "LuZ": 0.5, "EuZ": 0.5},
        "linear.fit.max.delta.depth.optics": {"Ed0": nan, "EdZ": 3.0, "LuZ": 2.5, "EuZ": 2.5},
        "depth.discretization": [0, 0.01, 1, 0.02, 2, 0.05, 5, 0.1, 10, 0.2, 20, 0.5, 50, 1, 100, 2, 200, 5, 500],
    }


def test_process_cast_fits_every_present_instrument():
    ds = _make_dataset(include_edz=True)
    result = process_cast(ds, _make_init())

    assert set(result.instrument_fits) == {"EdZ", "LuZ"}
    assert "EuZ" not in result.instrument_fits


def test_process_cast_skips_absent_instrument():
    ds = _make_dataset(include_edz=False)
    result = process_cast(ds, _make_init())

    assert set(result.instrument_fits) == {"LuZ"}


def test_process_cast_computes_rrs_when_luz_present():
    ds = _make_dataset()
    result = process_cast(ds, _make_init())

    assert result.rrs_loess is not None
    assert result.rrs_linear is not None
    # gentler-attenuation wavelengths should give finite, positive Rrs
    assert np.all(result.rrs_linear.rrs_0p[2:] > 0)
    assert np.all(np.isfinite(result.rrs_linear.rrs_0p[2:]))


def test_process_cast_rrs_none_without_luz():
    ds = xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), np.full((300, 4), 100.0)),
            "EdZ": (("time", "wavelength"), np.full((300, 4), 50.0)),
            "Ed0_Roll": ("time", np.zeros(300)),
            "Ed0_Pitch": ("time", np.zeros(300)),
            "EdZ_Roll": ("time", np.zeros(300)),
            "EdZ_Pitch": ("time", np.zeros(300)),
            "LuZ_Depth": ("time", np.linspace(0.05, 6.0, 300)),
        },
        coords={"time": np.arange(300), "wavelength": np.array(WAVES)},
    )

    result = process_cast(ds, _make_init())

    assert result.rrs_loess is None
    assert result.rrs_linear is None
    assert set(result.instrument_fits) == {"EdZ"}


def test_process_cast_recommends_linear_by_default():
    ds = _make_dataset()
    result = process_cast(ds, _make_init())

    assert result.rrs_method is None
    np.testing.assert_array_equal(result.recommended_rrs.rrs_0p, result.rrs_linear.rrs_0p)


def test_process_cast_recommends_loess_when_select_cops_dat_says_so():
    ds = _make_dataset()
    ds.attrs["rrs_method"] = "Rrs.0p"
    result = process_cast(ds, _make_init())

    assert result.rrs_method == "Rrs.0p"
    np.testing.assert_array_equal(result.recommended_rrs.rrs_0p, result.rrs_loess.rrs_0p)


def test_process_cast_recommends_linear_when_select_cops_dat_says_so():
    ds = _make_dataset()
    ds.attrs["rrs_method"] = "Rrs.0p.linear"
    result = process_cast(ds, _make_init())

    np.testing.assert_array_equal(result.recommended_rrs.rrs_0p, result.rrs_linear.rrs_0p)


def test_process_cast_recommended_rrs_falls_back_when_preferred_is_all_nan():
    # real-data finding (BoueesIML IML4_CAST_001, chl=2): select.cops.dat can
    # recommend "Rrs.0p.linear" for a station in general, but the linear
    # surface fit's R2/KS gate can still reject every wavelength for one
    # particular cast -- recommended_rrs must not return that all-NaN result
    # just because it isn't a bare None.
    ds = _make_dataset()
    ds.attrs["rrs_method"] = "Rrs.0p.linear"
    init = _make_init()
    init["linear.fit.Rsquared.threshold.optics"]["LuZ"] = 2.0  # R2 <= 1 always, so every window fails

    result = process_cast(ds, init)

    assert np.all(np.isnan(result.rrs_linear.rrs_0p))  # confirms the fixture actually forces the failure
    assert np.any(np.isfinite(result.rrs_loess.rrs_0p))
    np.testing.assert_array_equal(result.recommended_rrs.rrs_0p, result.rrs_loess.rrs_0p)


def test_process_cast_recommended_rrs_none_without_luz():
    ds = _make_dataset(include_edz=True)
    ds = ds.drop_vars("LuZ")
    ds.attrs["rrs_method"] = "Rrs.0p"
    result = process_cast(ds, _make_init())

    assert result.recommended_rrs is None


def test_process_cast_rrs_linear_matches_manual_compute_rrs():
    ds = _make_dataset()
    init = _make_init()
    result = process_cast(ds, init)

    expected = compute_rrs(
        result.instrument_fits["LuZ"].surface_linear.value_at_surface,
        result.ed0_fit.value_at_0,
        init["indice.water"],
        init["rau.Fresnel"],
    )
    np.testing.assert_allclose(result.rrs_linear.rrs_0p, expected.rrs_0p, equal_nan=True)


def _with_real_time(ds):
    times = pd.date_range("2019-08-18T18:18:00", periods=ds.sizes["time"], freq="s")
    return ds.assign_coords(time=times)


def test_process_cast_skips_shadow_correction_without_position_info():
    ds = _with_real_time(_make_dataset())
    result = process_cast(ds, _make_init())

    assert result.shadow_corrections == {}
    assert result.shadow_correction_note is not None


def test_process_cast_skips_shadow_correction_when_chl_is_nan():
    ds = _with_real_time(_make_dataset())
    ds.attrs["chl_flag"] = float("nan")
    ds.attrs["longitude"] = -68.108833
    ds.attrs["latitude"] = 49.13445
    result = process_cast(ds, _make_init())

    assert result.shadow_corrections == {}
    assert "NA" in result.shadow_correction_note or "chl" in result.shadow_correction_note.lower()


def test_process_cast_applies_shadow_correction_when_chl_999():
    ds = _with_real_time(_make_dataset())
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.108833
    ds.attrs["latitude"] = 49.13445

    result = process_cast(ds, _make_init())

    assert result.shadow_correction_note is None
    assert "LuZ" in result.shadow_corrections

    luz_fit = result.instrument_fits["LuZ"]
    corrected = luz_fit.surface_linear.value_at_surface / result.shadow_corrections["LuZ"].correction
    expected = compute_rrs(corrected, result.ed0_fit.value_at_0, 1.34, 0.043)
    np.testing.assert_allclose(result.rrs_linear.rrs_0p, expected.rrs_0p, equal_nan=True)
    # shadow correction should actually change the Rrs vs. the uncorrected fit
    uncorrected = compute_rrs(
        luz_fit.surface_linear.value_at_surface, result.ed0_fit.value_at_0, 1.34, 0.043
    )
    finite = np.isfinite(result.rrs_linear.rrs_0p) & np.isfinite(uncorrected.rrs_0p)
    assert finite.any()
    assert not np.allclose(result.rrs_linear.rrs_0p[finite], uncorrected.rrs_0p[finite])


def test_process_cast_shadow_correction_chl_zero_needs_absorption_table():
    ds = _with_real_time(_make_dataset())
    ds.attrs["chl_flag"] = 0.0
    ds.attrs["longitude"] = -68.108833
    ds.attrs["latitude"] = 49.13445

    result = process_cast(ds, _make_init())  # no absorption_waves/values passed

    assert result.shadow_corrections == {}
    assert result.shadow_correction_note is not None


def test_process_cast_uses_bioshade_split_when_given():
    ds = _with_real_time(_make_dataset())
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.108833
    ds.attrs["latitude"] = 49.13445

    waves = np.array(WAVES)
    ed0_dif = np.array([20.0, 18.0, 12.0, 6.0])
    ed0_tot = np.array([80.0, 70.0, 50.0, 25.0])
    bioshade = BioShadeResult(
        waves=waves, ed0_tot=ed0_tot, ed0_dif=ed0_dif, ed0_diffuse_fraction=ed0_dif / ed0_tot
    )

    result = process_cast(ds, _make_init(), bioshade=bioshade)
    without_bioshade = process_cast(ds, _make_init())

    assert result.shadow_correction_note is None
    assert "LuZ" in result.shadow_corrections
    np.testing.assert_allclose(result.shadow_corrections["LuZ"].edif, ed0_dif)
    np.testing.assert_allclose(result.shadow_corrections["LuZ"].edir, ed0_tot - ed0_dif)

    # the measured split must actually change the correction vs. the Gregg & Carder fallback
    finite = np.isfinite(result.rrs_linear.rrs_0p) & np.isfinite(without_bioshade.rrs_linear.rrs_0p)
    assert finite.any()
    assert not np.allclose(
        result.rrs_linear.rrs_0p[finite], without_bioshade.rrs_linear.rrs_0p[finite]
    )


def test_process_cast_position_override_enables_shadow_correction_without_attrs():
    # no ds.attrs["longitude"/"latitude"] at all -- the override alone must be enough
    ds = _with_real_time(_make_dataset())
    ds.attrs["chl_flag"] = 999.0

    result = process_cast(
        ds, _make_init(), position_override=PositionOverride(longitude=-68.108833, latitude=49.13445)
    )

    assert result.shadow_correction_note is None
    assert "LuZ" in result.shadow_corrections


def test_process_cast_position_override_wins_over_attrs():
    ds = _with_real_time(_make_dataset())
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.108833
    ds.attrs["latitude"] = 49.13445

    with_attrs = process_cast(ds, _make_init())
    overridden = process_cast(
        ds, _make_init(), position_override=PositionOverride(longitude=100.0, latitude=-30.0)
    )

    finite = np.isfinite(with_attrs.rrs_linear.rrs_0p) & np.isfinite(overridden.rrs_linear.rrs_0p)
    assert finite.any()
    assert not np.allclose(with_attrs.rrs_linear.rrs_0p[finite], overridden.rrs_linear.rrs_0p[finite])


def test_process_cast_utc_time_override_changes_sun_geometry():
    ds = _with_real_time(_make_dataset())
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.108833
    ds.attrs["latitude"] = 49.13445

    normal = process_cast(ds, _make_init())
    shifted = process_cast(
        ds,
        _make_init(),
        position_override=PositionOverride(utc_time=pd.Timestamp("2019-01-18T04:18:00")),
    )

    finite = np.isfinite(normal.rrs_linear.rrs_0p) & np.isfinite(shifted.rrs_linear.rrs_0p)
    assert finite.any()
    assert not np.allclose(normal.rrs_linear.rrs_0p[finite], shifted.rrs_linear.rrs_0p[finite])


def test_process_cast_resolved_position_matches_attrs_when_no_override():
    ds = _with_real_time(_make_dataset())
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.108833
    ds.attrs["latitude"] = 49.13445

    result = process_cast(ds, _make_init())

    assert result.resolved_longitude == -68.108833
    assert result.resolved_latitude == 49.13445


def test_process_cast_resolved_position_reflects_override_not_attrs():
    ds = _with_real_time(_make_dataset())
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.108833
    ds.attrs["latitude"] = 49.13445

    result = process_cast(
        ds, _make_init(), position_override=PositionOverride(longitude=10.0, latitude=-30.0)
    )

    assert result.resolved_longitude == 10.0
    assert result.resolved_latitude == -30.0


def test_process_cast_resolved_position_none_when_shadow_correction_skipped():
    ds = _with_real_time(_make_dataset())  # no chl_flag/position attrs at all

    result = process_cast(ds, _make_init())

    assert result.resolved_longitude is None
    assert result.resolved_latitude is None


def _make_euz_dataset(n=300, ed0_level=100.0):
    sensor_depth = np.linspace(0.05, 6.0, n)
    waves = np.array(WAVES)
    K = np.array(K_TRUE)

    euz_true_depth = sensor_depth + DELTA_CAPTEUR_LUZ  # reuse an arbitrary offset for EuZ too
    euz = np.array(LUZ_X0_TRUE)[None, :] * np.exp(-K[None, :] * euz_true_depth[:, None])
    edz_true_depth = sensor_depth + DELTA_CAPTEUR_EDZ
    edz = np.array(EDZ_X0_TRUE)[None, :] * np.exp(-K[None, :] * edz_true_depth[:, None])

    ed0 = np.full((n, len(waves)), ed0_level)
    zeros = np.zeros(n)

    times = pd.date_range("2019-08-18T18:18:00", periods=n, freq="s")
    return xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), ed0),
            "EuZ": (("time", "wavelength"), euz),
            "EdZ": (("time", "wavelength"), edz),
            "Ed0_Roll": ("time", zeros),
            "Ed0_Pitch": ("time", zeros),
            "EdZ_Roll": ("time", zeros),
            "EdZ_Pitch": ("time", zeros),
            "EuZ_Roll": ("time", zeros),
            "EuZ_Pitch": ("time", zeros),
            "EdZ_Depth": ("time", sensor_depth),
            "EuZ_Temp": ("time", np.full(n, 10.0)),
        },
        coords={"time": times, "wavelength": waves},
    )


def _make_init_euz_only():
    init = _make_init()
    init["depth.is.on"] = "EdZ"
    return init


def test_process_cast_computes_rrs_from_euz_when_no_luz():
    ds = _make_euz_dataset()
    result = process_cast(ds, _make_init_euz_only())

    assert set(result.instrument_fits) == {"EdZ", "EuZ"}
    assert result.rrs_source == "EuZ"
    assert result.rrs_loess is not None
    assert result.rrs_linear is not None
    assert np.all(result.rrs_linear.rrs_0p[2:] > 0)
    assert np.all(np.isfinite(result.rrs_linear.rrs_0p[2:]))


def test_process_cast_euz_derived_rrs_matches_manual_q_factor_division():
    ds = _make_euz_dataset()
    result = process_cast(ds, _make_init_euz_only())

    euz_fit = result.instrument_fits["EuZ"]
    expected = compute_rrs(
        euz_fit.surface_linear.value_at_surface / np.pi,
        result.ed0_fit.value_at_0,
        1.34,
        0.043,
    )
    np.testing.assert_allclose(result.rrs_linear.rrs_0p, expected.rrs_0p, equal_nan=True)


def test_process_cast_rrs_source_is_luz_when_both_present():
    ds = _make_euz_dataset()
    ds["LuZ"] = ds["EuZ"]  # both present -- LuZ must still win
    result = process_cast(ds, _make_init_euz_only())

    assert result.rrs_source == "LuZ"


def test_process_cast_euz_only_shadow_corrected_before_q_division():
    ds = _with_real_time(_make_euz_dataset())
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.108833
    ds.attrs["latitude"] = 49.13445

    result = process_cast(ds, _make_init_euz_only())

    assert result.rrs_source == "EuZ"
    assert "EuZ" in result.shadow_corrections
    euz_fit = result.instrument_fits["EuZ"]
    euz_shadow = result.shadow_corrections["EuZ"]
    corrected = (euz_fit.surface_linear.value_at_surface / euz_shadow.correction) / np.pi
    expected = compute_rrs(corrected, result.ed0_fit.value_at_0, 1.34, 0.043)
    np.testing.assert_allclose(result.rrs_linear.rrs_0p, expected.rrs_0p, equal_nan=True)


def test_process_cast_euz_only_positive_chl_leaves_rrs_none():
    ds = _make_euz_dataset()
    ds.attrs["chl_flag"] = 2.5  # real chlorophyll concentration -- Q factor not ported

    result = process_cast(ds, _make_init_euz_only())

    assert result.rrs_source is None
    assert result.rrs_loess is None
    assert result.rrs_linear is None
    assert set(result.instrument_fits) == {"EdZ", "EuZ"}  # fitting itself is unaffected


def test_process_cast_computes_nlw_when_bandwidth_present():
    ds = _make_dataset()
    init = _make_init()
    init["bandwidth"] = 10.0

    result = process_cast(ds, init)

    assert result.rrs_linear.nlw_0p is not None
    assert result.rrs_linear.nlw_0p.shape == np.array(WAVES).shape


def test_process_cast_nlw_none_without_bandwidth():
    ds = _make_dataset()
    result = process_cast(ds, _make_init())  # no "bandwidth" key

    assert result.rrs_linear.nlw_0p is None


def test_process_cast_computes_qwip_diagnostics():
    ds = _make_dataset()
    init = _make_init()
    init["bandwidth"] = 10.0

    result = process_cast(ds, init)

    assert result.qwip_linear is not None
    assert np.isfinite(result.qwip_linear.avw)
    assert 1 <= result.qwip_linear.fu <= 21


def test_process_cast_qwip_none_without_rrs():
    ds = xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), np.full((300, 4), 100.0)),
            "EdZ": (("time", "wavelength"), np.full((300, 4), 50.0)),
            "Ed0_Roll": ("time", np.zeros(300)),
            "Ed0_Pitch": ("time", np.zeros(300)),
            "EdZ_Roll": ("time", np.zeros(300)),
            "EdZ_Pitch": ("time", np.zeros(300)),
            "LuZ_Depth": ("time", np.linspace(0.05, 6.0, 300)),
        },
        coords={"time": np.arange(300), "wavelength": np.array(WAVES)},
    )

    result = process_cast(ds, _make_init())

    assert result.qwip_loess is None
    assert result.qwip_linear is None
