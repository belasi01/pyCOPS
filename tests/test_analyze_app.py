from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from conftest import _write_cast_file  # noqa: E402
from pycops.io.config import CastInfo, write_init_cops  # noqa: E402
from pycops.io.netcdf import write_cast_result  # noqa: E402
from pycops.processing.process_cast import process_cast  # noqa: E402
from pycops.ui.analyze_app import (  # noqa: E402
    _effective_tiltmax,
    _effective_time_window,
    _k0_at_adaptive_depth,
    _mask_negligible_rb,
    _raw_scan_values,
    _visible_band_ylim,
)

_APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "pycops" / "ui" / "clean_app.py")

_CAST_FILE = "WISE_CAST_001_190817_220856_URC.csv"
_CAST_STEM = "WISE_CAST_001_190817_220856_URC"

WAVES = (340.0, 380.0, 443.0, 555.0)
K_TRUE = (2.0, 0.9, 0.3, 0.1)
LUZ_X0 = (0.0015, 0.006, 0.05, 0.3)
EDZ_X0 = (50.0, 80.0, 150.0, 300.0)
EUZ_X0 = (0.002, 0.008, 0.06, 0.35)
DELTA_LUZ = 0.238
DELTA_EDZ = -0.05
DELTA_EUZ = 0.238


def _make_full_dataset(n=300):
    """EdZ+LuZ+EuZ, a real position/chl (shadow correction) and shallow=True (bottom
    reflectance) -- exercises every optional section render_analyze_tab() can show."""
    depth = np.linspace(0.05, 8.0, n)
    waves = np.array(WAVES)
    K = np.array(K_TRUE)
    luz = np.array(LUZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_LUZ)[:, None])
    edz = np.array(EDZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_EDZ)[:, None])
    euz = np.array(EUZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_EUZ)[:, None])
    ed0 = np.full((n, len(waves)), 100.0)
    zeros = np.zeros(n)
    times = pd.date_range("2019-08-17T22:08:56", periods=n, freq="s")
    ds = xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), ed0),
            "LuZ": (("time", "wavelength"), luz),
            "EdZ": (("time", "wavelength"), edz),
            "EuZ": (("time", "wavelength"), euz),
            "Ed0_Roll": ("time", zeros),
            "Ed0_Pitch": ("time", zeros),
            "EdZ_Roll": ("time", zeros),
            "EdZ_Pitch": ("time", zeros),
            "EuZ_Roll": ("time", zeros),
            "EuZ_Pitch": ("time", zeros),
            "LuZ_Depth": ("time", depth),
            "EuZ_Depth": ("time", depth),
            "LuZ_Temp": ("time", np.full(n, 10.0)),
            "EuZ_Temp": ("time", np.full(n, 10.0)),
        },
        coords={"time": times, "wavelength": waves},
    )
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.11626
    ds.attrs["latitude"] = 49.24872
    ds.attrs["shallow"] = True
    ds.attrs["rrs_method"] = "Rrs.0p.linear"
    return ds


def _make_full_init():
    nan = float("nan")
    return {
        "depth.is.on": "LuZ",
        "indice.water": 1.34,
        "rau.Fresnel": 0.043,
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 7.0, "LuZ": 7.0, "EuZ": 7.0},
        "radius.instrument.optics": {"Ed0": 0.035, "EdZ": 0.035, "LuZ": 0.035, "EuZ": 0.035},
        "delta.capteur.optics": {"Ed0": 0.0, "EdZ": DELTA_EDZ, "LuZ": DELTA_LUZ, "EuZ": DELTA_EUZ},
        "sub.surface.removed.layer.optics": {"Ed0": 0.0, "EdZ": 0.3, "LuZ": 0.0, "EuZ": 0.0},
        "depth.interval.for.smoothing.optics": {"Ed0": 10.0, "EdZ": 3.0, "LuZ": 3.0, "EuZ": 3.0},
        "linear.fit.Rsquared.threshold.optics": {"Ed0": nan, "EdZ": 0.5, "LuZ": 0.5, "EuZ": 0.5},
        "linear.fit.max.delta.depth.optics": {"Ed0": nan, "EdZ": 3.0, "LuZ": 2.5, "EuZ": 2.5},
        "depth.discretization": [0, 0.01, 1, 0.02, 2, 0.05, 5, 0.1, 10, 0.2, 20, 0.5, 50, 1, 100, 2, 200, 5, 500],
        "bandwidth": 10.0,
    }


def _make_minimal_dataset(n=200):
    """LuZ+EdZ only, no position/chl (no shadow correction), not shallow (no bottom)."""
    depth = np.linspace(0.05, 5.0, n)
    waves = np.array(WAVES)
    K = np.array(K_TRUE)
    luz = np.array(LUZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_LUZ)[:, None])
    edz = np.array(EDZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_EDZ)[:, None])
    ed0 = np.full((n, len(waves)), 100.0)
    zeros = np.zeros(n)
    times = pd.date_range("2019-08-17T22:08:56", periods=n, freq="s")
    return xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), ed0),
            "LuZ": (("time", "wavelength"), luz),
            "EdZ": (("time", "wavelength"), edz),
            "Ed0_Roll": ("time", zeros),
            "Ed0_Pitch": ("time", zeros),
            "EdZ_Roll": ("time", zeros),
            "EdZ_Pitch": ("time", zeros),
            "LuZ_Depth": ("time", depth),
            "LuZ_Temp": ("time", np.full(n, 10.0)),
        },
        coords={"time": times, "wavelength": waves},
    )


def _make_minimal_init():
    nan = float("nan")
    return {
        "depth.is.on": "LuZ",
        "indice.water": 1.34,
        "rau.Fresnel": 0.043,
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 7.0, "LuZ": 7.0},
        "radius.instrument.optics": {"Ed0": 0.035, "EdZ": 0.035, "LuZ": 0.035},
        "delta.capteur.optics": {"Ed0": 0.0, "EdZ": DELTA_EDZ, "LuZ": DELTA_LUZ},
        "sub.surface.removed.layer.optics": {"Ed0": 0.0, "EdZ": 0.3, "LuZ": 0.0},
        "depth.interval.for.smoothing.optics": {"Ed0": 10.0, "EdZ": 3.0, "LuZ": 3.0},
        "linear.fit.Rsquared.threshold.optics": {"Ed0": nan, "EdZ": 0.5, "LuZ": 0.5},
        "linear.fit.max.delta.depth.optics": {"Ed0": nan, "EdZ": 3.0, "LuZ": 2.5},
        "depth.discretization": [0, 0.01, 1, 0.02, 2, 0.05, 5, 0.1, 10, 0.2, 20, 0.5, 50, 1, 100, 2, 200, 5, 500],
        "bandwidth": 10.0,
    }


def _write_nc(tmp_path, ds, init, filename=_CAST_FILE, stem=_CAST_STEM):
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir(parents=True, exist_ok=True)
    result = process_cast(ds, init)
    write_cast_result(result, nc_dir / f"{stem}.nc", ds=ds)

    # A real init.cops.dat on disk too (not just the in-memory `init` dict process_cast() took),
    # matching real usage -- render_analyze_tab()'s "Adjust & reprocess" section needs to read
    # depth_is_on from an actual file.
    init_for_file = dict(init)
    init_for_file.setdefault("instruments.optics", list(init["tiltmax.optics"].keys()))
    init_for_file.setdefault("number.of.fields.before.date", 3.0)
    write_init_cops(tmp_path / "init.cops.dat", init_for_file, overwrite=True)
    return result


def test_analyze_tab_missing_nc_folder_shows_error(tmp_path):
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert any("nc/ subfolder" in e.value for e in at.error)


def test_analyze_tab_full_cast_renders_every_section(tmp_path):
    _write_nc(tmp_path, _make_full_dataset(), _make_full_init())
    _write_cast_file(tmp_path, _CAST_FILE)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=60)

    assert not at.exception
    assert at.selectbox(key="analyze_cast_select").options == [_CAST_STEM]

    # Overview + Ed0 stability render unconditionally.
    assert any("Overview" in s.value for s in at.subheader)
    assert any("Ed0 stability" in s.value for s in at.subheader)

    # All three depth-profiled instruments got their own expanders.
    expander_labels = [e.label for e in at.expander]
    for instrument in ("EdZ", "LuZ", "EuZ"):
        assert f"{instrument} depth profile" in expander_labels
        assert f"{instrument} attenuation (K)" in expander_labels

    # Exercise the wavelength drill-down (raw-scan overlay code path) on LuZ -- also confirms the
    # EdZ raw-scan overlay fix (both LuZ and EdZ use depth_is_on's own depth column, not
    # "{instrument}_Depth", which doesn't exist for EdZ).
    at.selectbox(key="analyze_LuZ_depth_wave").set_value("340").run(timeout=30)
    assert not at.exception
    at.selectbox(key="analyze_EdZ_depth_wave").set_value("340").run(timeout=30)
    assert not at.exception

    # Extrapolation-comparison sections (LOESS vs. linear) for LuZ/EuZ.
    assert "LuZ extrapolation methods (LOESS vs. linear)" in expander_labels
    assert "EuZ extrapolation methods (LOESS vs. linear)" in expander_labels
    at.selectbox(key="analyze_LuZ_extrap_wave").set_value("340").run(timeout=30)
    assert not at.exception

    assert any("Rrs" in s.value for s in at.subheader)
    assert any("Recommended" in c.value for c in at.caption)
    assert "LuZ shadow correction" in expander_labels
    assert "EuZ shadow correction" in expander_labels
    assert any("QWIP" in s.value for s in at.subheader)
    assert "LuZ bottom reflectance" in expander_labels
    assert "EuZ bottom reflectance" in expander_labels
    assert any("Benthic PAR" in m.label for m in at.metric)

    # PAR & Kd(PAR) section: renders, including the Kd(PAR)-vs-depth plot (no exception).
    assert "PAR & Kd(PAR)" in expander_labels


def test_analyze_tab_minimal_cast_skips_optional_sections(tmp_path):
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    expander_labels = [e.label for e in at.expander]
    assert "EuZ depth profile" not in expander_labels
    assert "LuZ shadow correction" not in expander_labels
    assert "LuZ bottom reflectance" not in expander_labels
    # QWIP still renders even without shadow correction -- Rrs falls back to the uncorrected
    # LuZ surface value rather than being skipped entirely, so QWIP still has something to score.


def test_analyze_tab_missing_raw_file_still_renders_nc_content(tmp_path):
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    # No raw cast file written alongside nc/.

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert any("raw cast file not found" in c.value for c in at.caption)
    assert any("Overview" in s.value for s in at.subheader)


def test_raw_scan_values_applies_delta_capteur_offset():
    """Regression test: the raw depth column is the reference sensor's own depth (depth_is_on),
    not this instrument's true depth -- delta_capteur_optics (the sensor-to-sensor offset already
    applied by cast_fit.py before fitting) must be added, or the raw scatter and the fitted curve
    end up systematically offset from each other (Simon: EdZ's fit sat above its raw points, LuZ's
    below -- exactly what a missing, oppositely-signed offset per instrument would cause)."""
    raw_ds = xr.Dataset(
        {"LuZ": (("time", "wavelength"), np.array([[1.0], [2.0]])), "LuZ_Depth": ("time", np.array([1.0, 2.0]))},
        coords={"wavelength": [340.0]},
    )
    values, depth = _raw_scan_values(raw_ds, "LuZ", 0.238, "LuZ", 340.0)
    np.testing.assert_allclose(depth, [1.238, 2.238])

    values, depth = _raw_scan_values(raw_ds, "LuZ", None, "LuZ", 340.0)
    np.testing.assert_allclose(depth, [1.0, 2.0])


def test_mask_negligible_rb_flags_near_zero_denominator_and_nan():
    rb = np.array([1.0, 2.0, 3.0, 4.0])
    rb_extrapolated = np.array([1.1, 2.1, 3.1, 4.1])
    edz_surface = np.array([100.0, 100.0, 100.0, 100.0])
    # wave 0: NaN at bottom (fast-attenuating channel with no valid fitted point that deep);
    # wave 1: 0.5% of surface (below the 1% threshold); wave 2/3: comfortably above it.
    edz_bottom = np.array([np.nan, 0.5, 9.0, 45.0])

    masked_rb, masked_rb_extrap = _mask_negligible_rb(rb, rb_extrapolated, edz_bottom, edz_surface)

    assert np.isnan(masked_rb[0]) and np.isnan(masked_rb_extrap[0])
    assert np.isnan(masked_rb[1]) and np.isnan(masked_rb_extrap[1])
    assert masked_rb[2] == 3.0 and masked_rb[3] == 4.0
    assert masked_rb_extrap[2] == 3.1 and masked_rb_extrap[3] == 4.1
    # the inputs aren't mutated in place
    assert not np.isnan(rb[0])


def test_visible_band_ylim_ignores_nir_fluorescence_spike():
    waves = np.array([443.0, 555.0, 683.0, 780.0])
    rb = np.array([0.05, 0.08, 0.06, 1.4])  # 780 nm: >100%, a fluorescence artifact
    rb_extrapolated = np.array([0.06, 0.09, 0.07, 1.5])

    ylim = _visible_band_ylim(rb, rb_extrapolated, waves)

    assert ylim is not None
    assert ylim < 1.0  # scaled from the <=700 nm bands only, not the 780 nm spike


def test_visible_band_ylim_none_when_nothing_finite():
    waves = np.array([443.0, 555.0])
    nan_array = np.full(2, np.nan)
    assert _visible_band_ylim(nan_array, nan_array, waves) is None


def test_analyze_tab_discard_button_sets_flag_rejected(tmp_path):
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)
    at.button(key="analyze_discard").click().run(timeout=30)

    assert not at.exception
    select_text = (tmp_path / "select.cops.dat").read_text()
    assert f"{_CAST_FILE};0;" in select_text
    assert any(f"Discarded {_CAST_FILE}" in s.value and "select.cops.dat" in s.value for s in at.success)


def test_analyze_tab_validate_and_next_sets_flag_normal_and_advances(tmp_path):
    second_stem = "WISE_CAST_002_190817_221224_URC"
    second_file = f"{second_stem}.csv"
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init(), stem=_CAST_STEM)
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init(), stem=second_stem)
    _write_cast_file(tmp_path, _CAST_FILE)
    _write_cast_file(tmp_path, second_file)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)
    assert at.selectbox(key="analyze_cast_select").value == _CAST_STEM

    at.button(key="analyze_validate_next").click().run(timeout=30)

    assert not at.exception
    assert at.selectbox(key="analyze_cast_select").value == second_stem
    select_text = (tmp_path / "select.cops.dat").read_text()
    assert f"{_CAST_FILE};1;" in select_text
    assert any(f"Validated {_CAST_FILE}" in s.value and "select.cops.dat" in s.value for s in at.success)


def test_analyze_tab_validate_and_next_disabled_when_method_changed_but_not_reprocessed(tmp_path):
    """Regression test for a real bug Simon reported: picked LOESS, checked a wavelength to
    exclude, then clicked 'Validate and next' directly (skipping Reprocess) -- the method showed
    as saved on reload, but the Rrs value was never actually NaN'd, because Validate/Discard used
    to silently persist the live (unreprocessed) method while dropping everything else. Changing
    the method widget without reprocessing must now disable Validate and warn, not silently save."""
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)
    (tmp_path / "select.cops.dat").write_text(f"{_CAST_FILE};1;Rrs.0p.linear;NA\n")

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)
    at.selectbox(key=f"analyze_method::{_CAST_STEM}").set_value("Rrs.0p").run(timeout=30)

    assert not at.exception
    assert at.button(key="analyze_validate_next").disabled
    assert any("Reprocess with adjusted parameters" in w.value for w in at.warning)


def test_analyze_tab_discard_preserves_saved_method_even_if_widget_changed(tmp_path):
    """Even Discard (never gated on unsaved changes, since a discarded cast's Rrs doesn't matter)
    must not silently adopt the live, unreprocessed method choice -- select.cops.dat's method
    column must stay in sync with what the .nc actually reflects."""
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)
    (tmp_path / "select.cops.dat").write_text(f"{_CAST_FILE};1;Rrs.0p.linear;NA\n")

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)
    at.selectbox(key=f"analyze_method::{_CAST_STEM}").set_value("Rrs.0p").run(timeout=30)
    at.button(key="analyze_discard").click().run(timeout=30)

    assert not at.exception
    select_text = (tmp_path / "select.cops.dat").read_text()
    assert f"{_CAST_FILE};0;Rrs.0p.linear;" in select_text  # still the saved method, not "Rrs.0p"


def test_analyze_tab_validate_and_next_enabled_again_after_reprocess(tmp_path, monkeypatch):
    """Once Reprocess has actually applied a changed method, Validate and next must re-enable."""
    import pycops.ui.analyze_app as analyze_app_module
    from pycops.processing.deployment import ReprocessedCast

    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)
    (tmp_path / "select.cops.dat").write_text(f"{_CAST_FILE};1;Rrs.0p.linear;NA\n")

    fake_result = process_cast(_make_minimal_dataset(), _make_minimal_init())
    monkeypatch.setattr(
        analyze_app_module,
        "reprocess_single_cast",
        lambda directory, file, position_overrides=None: ReprocessedCast(
            result=fake_result, ds=_make_minimal_dataset()
        ),
    )

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)
    at.selectbox(key=f"analyze_method::{_CAST_STEM}").set_value("Rrs.0p").run(timeout=30)
    assert at.button(key="analyze_validate_next").disabled

    at.button(key="analyze_reprocess").click().run(timeout=30)

    assert not at.exception
    assert not at.button(key="analyze_validate_next").disabled


def test_analyze_tab_reprocess_button_saves_overrides_and_rewrites_nc(tmp_path, monkeypatch):
    """The Reprocess button must (1) actually persist the typed override to info.cops.dat and
    (2) overwrite the cast's .nc with whatever reprocess_single_cast() returns -- verified with a
    monkeypatched reprocess_single_cast (patched on pycops.ui.analyze_app, the module that
    actually calls it) so this test doesn't depend on the tiny synthetic raw file being large
    enough for a real re-fit to succeed."""
    import pycops.ui.analyze_app as analyze_app_module
    from pycops.processing.deployment import ReprocessedCast

    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)

    fake_ds = _make_minimal_dataset()
    fake_ds["Ed0"] = fake_ds["Ed0"] * 5.0  # distinguishable from the original's ed0_level=100.0
    fake_result = process_cast(fake_ds, _make_minimal_init())
    monkeypatch.setattr(
        analyze_app_module,
        "reprocess_single_cast",
        lambda directory, file, position_overrides=None: ReprocessedCast(result=fake_result, ds=fake_ds),
    )

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)
    at.text_input(key=f"analyze_override::{_CAST_STEM}_tiltmax").set_value("2,2,2").run(timeout=30)
    at.button(key="analyze_reprocess").click().run(timeout=30)

    assert not at.exception
    info_text = (tmp_path / "info.cops.dat").read_text()
    assert "2,2,2" in info_text

    nc = xr.open_dataset(tmp_path / "nc" / f"{_CAST_STEM}.nc")
    np.testing.assert_allclose(nc["ed0_value_at_0"].values, fake_result.ed0_fit.value_at_0)
    nc.close()


def test_analyze_tab_method_selectbox_defaults_to_existing_select_cops_dat(tmp_path):
    """The Rrs method dropdown must pre-fill from whatever select.cops.dat already records for
    this cast, not always the module default -- same pre-fill contract as the other Adjust &
    reprocess widgets (time.window, overrides)."""
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)
    (tmp_path / "select.cops.dat").write_text(f"{_CAST_FILE};1;Rrs.0p;NA\n")

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert at.selectbox(key=f"analyze_method::{_CAST_STEM}").value == "Rrs.0p"


def test_analyze_tab_reprocess_button_saves_method_to_select_cops_dat(tmp_path, monkeypatch):
    """Simon's request: the Rrs extrapolation method (select.cops.dat's method column) must be
    editable inside Adjust & reprocess, and a changed choice must persist there -- kept in
    select.cops.dat rather than moved elsewhere, since that's the R package's own long-established
    location and update_cast_selection() already tolerates short/legacy rows."""
    import pycops.ui.analyze_app as analyze_app_module
    from pycops.processing.deployment import ReprocessedCast

    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)
    (tmp_path / "select.cops.dat").write_text(f"{_CAST_FILE};1;Rrs.0p.linear;NA\n")

    fake_result = process_cast(_make_minimal_dataset(), _make_minimal_init())
    monkeypatch.setattr(
        analyze_app_module,
        "reprocess_single_cast",
        lambda directory, file, position_overrides=None: ReprocessedCast(
            result=fake_result, ds=_make_minimal_dataset()
        ),
    )

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)
    at.selectbox(key=f"analyze_method::{_CAST_STEM}").set_value("Rrs.0p").run(timeout=30)
    at.button(key="analyze_reprocess").click().run(timeout=30)

    assert not at.exception
    select_text = (tmp_path / "select.cops.dat").read_text()
    assert f"{_CAST_FILE};1;Rrs.0p;NA" in select_text  # method changed, flag/shallow preserved


def test_wavelength_exclusion_table_prechecks_existing_exclusions():
    from pycops.ui.analyze_app import _wavelength_exclusion_table

    nc = xr.Dataset(
        {
            "rrs_0p_loess": ("wavelength", [1.0, 2.0, 3.0]),
            "rrs_0p_linear": ("wavelength", [1.1, 2.1, 3.1]),
        },
        coords={"wavelength": [340.0, 380.0, 443.0]},
    )

    table = _wavelength_exclusion_table(nc, existing=[380.0])

    assert list(table["Exclude (set to NaN)"]) == [False, True, False]
    assert list(table["Wavelength (nm)"]) == [340.0, 380.0, 443.0]


def test_wavelength_exclusion_table_no_existing_exclusions_all_unchecked():
    from pycops.ui.analyze_app import _wavelength_exclusion_table

    nc = xr.Dataset(
        {"rrs_0p_loess": ("wavelength", [1.0, 2.0])},
        coords={"wavelength": [340.0, 380.0]},
    )

    table = _wavelength_exclusion_table(nc, existing=[])

    assert list(table["Exclude (set to NaN)"]) == [False, False]


def test_analyze_tab_wavelength_exclusion_editor_renders(tmp_path):
    """Smoke test only -- st.data_editor isn't yet interactable via Streamlit's AppTest, so
    checkbox-toggle-then-reprocess is validated in real-browser smoke testing instead (see
    CLAUDE.md); this confirms the section renders and pre-fills from a saved sidecar entry
    without exception."""
    from pycops.io.exclusions import update_wavelength_exclusions

    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)
    update_wavelength_exclusions(tmp_path / "rrs_wavelength_exclusions.cops.dat", _CAST_FILE, [380.0])

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert any("wavelength exclusions" in c.value for c in at.caption)


def test_analyze_tab_qwip_shallow_water_shows_note_instead_of_failed(tmp_path):
    """A shallow cast whose QWIP score fails (|score|>=0.1) and is negative should read as
    expected/informational, not as a quality-control failure -- Simon's own domain-knowledge
    addition, since bottom-reflected light distorts the open-ocean-calibrated QWIP shape."""
    _write_nc(tmp_path, _make_full_dataset(), _make_full_init())
    _write_cast_file(tmp_path, _CAST_FILE)

    nc_path = tmp_path / "nc" / f"{_CAST_STEM}.nc"
    with xr.open_dataset(nc_path) as opened:
        nc = opened.load()
    nc.attrs["qwip_loess_score"] = -0.15
    nc.attrs["qwip_loess_passed"] = 0
    nc.to_netcdf(nc_path, mode="w")

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert any("Negative QWIP score expected" in m.value for m in at.markdown)
    assert not any("Failed" in m.value for m in at.markdown)


def test_k0_at_adaptive_depth_uses_per_wavelength_z_interval():
    depth_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    k0 = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=float)
    z_interval = np.array([1.0, 3.0])

    result = _k0_at_adaptive_depth(k0, depth_grid, z_interval)

    np.testing.assert_allclose(result, [2.0, 40.0])


def test_k0_at_adaptive_depth_falls_back_to_2m_when_any_z_interval_nan():
    """Port of plot.Rrs.Kd.for.station.R's all-or-nothing fallback: one invalid linear fit
    forces every wavelength (not just that one) onto the fixed ~2 m depth."""
    depth_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    k0 = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=float)
    z_interval = np.array([1.0, np.nan])

    result = _k0_at_adaptive_depth(k0, depth_grid, z_interval)

    np.testing.assert_allclose(result, [3.0, 30.0])  # depth nearest 2 m -> index 2, both bands




def test_analyze_tab_station_comparison_overlays_kept_casts(tmp_path):
    second_stem = "WISE_CAST_002_190817_221224_URC"
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init(), stem=_CAST_STEM)
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init(), stem=second_stem)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)
    at.radio(key="analyze_mode").set_value("Station comparison (Rrs & Kd)").run(timeout=30)

    assert not at.exception
    assert any("Rrs" in s.value for s in at.subheader)
    assert any("K0" in s.value for s in at.subheader)


def test_analyze_tab_station_comparison_warns_when_all_discarded(tmp_path):
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    (tmp_path / "select.cops.dat").write_text(f"{_CAST_FILE};0;Rrs.0p;NA\n")

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)
    at.radio(key="analyze_mode").set_value("Station comparison (Rrs & Kd)").run(timeout=30)

    assert not at.exception
    assert any("No kept casts" in w.value for w in at.warning)


def test_effective_tiltmax_uses_override_when_present():
    init = {
        "instruments.optics": ("Ed0", "EdZ", "LuZ"),
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 5.0, "LuZ": 5.0},
    }
    info = CastInfo(
        file="x",
        longitude=None,
        latitude=None,
        chl_flag=None,
        time_window=None,
        sub_surface_removed_layer=None,
        tiltmax=[10.0, 2.0, 3.0],
        depth_interval_for_smoothing=None,
        dark_files=[],
    )

    assert _effective_tiltmax(init, info, "EdZ") == 2.0


def test_effective_tiltmax_falls_back_to_init_default_when_no_override():
    init = {
        "instruments.optics": ("Ed0", "EdZ", "LuZ"),
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 5.0, "LuZ": 5.0},
    }

    assert _effective_tiltmax(init, None, "EdZ") == 5.0


def test_effective_time_window_uses_info_override_when_present():
    init = {"time.window": [0.0, 100.0]}
    info = CastInfo(
        file="x",
        longitude=None,
        latitude=None,
        chl_flag=None,
        time_window=(3.5, 22.5),
        sub_surface_removed_layer=None,
        tiltmax=None,
        depth_interval_for_smoothing=None,
        dark_files=[],
    )

    assert _effective_time_window(init, info) == (3.5, 22.5)


def test_effective_time_window_falls_back_to_init_default_when_no_override():
    init = {"time.window": [0.0, 100.0]}

    assert _effective_time_window(init, None) == (0.0, 100.0)


def test_effective_time_window_none_when_neither_set():
    assert _effective_time_window({}, None) is None


def test_analyze_tab_depth_vs_time_section_always_visible(tmp_path):
    """Simon's request: the depth_is_on-vs-elapsed-time diagnostic should be visible without
    opening 'Adjust & reprocess' -- it's what he looks at to decide a time.window trim."""
    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert any("depth vs time" in h.value for h in at.subheader)


def test_analyze_tab_reprocess_shows_success_banner_after_rerun(tmp_path, monkeypatch):
    """After Reprocess -> rerun, a banner should confirm the page reloaded with the new result --
    Simon reported not being sure a reprocess actually took effect."""
    import pycops.ui.analyze_app as analyze_app_module
    from pycops.processing.deployment import ReprocessedCast

    _write_nc(tmp_path, _make_minimal_dataset(), _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)

    fake_result = process_cast(_make_minimal_dataset(), _make_minimal_init())
    monkeypatch.setattr(
        analyze_app_module,
        "reprocess_single_cast",
        lambda directory, file, position_overrides=None: ReprocessedCast(
            result=fake_result, ds=_make_minimal_dataset()
        ),
    )

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)
    at.button(key="analyze_reprocess").click().run(timeout=30)

    assert not at.exception
    banner = next((s.value for s in at.success if f"Reprocessed {_CAST_FILE}" in s.value), None)
    assert banner is not None
    # Simon's request: confirm which files were actually saved, and that it's now safe to move on.
    assert "select.cops.dat" in banner
    assert "info.cops.dat" in banner
    assert ".nc" in banner
    assert "Validate and next" in banner


def test_analyze_tab_tilt_expanders_render_for_every_instrument(tmp_path):
    """LuZ has no Roll/Pitch of its own in the fixture (matching real deployments) -- confirms
    the fallback-to-EdZ's-inclinometer logic (tilt.py's own _TILT_FALLBACK) also works here."""
    _write_nc(tmp_path, _make_full_dataset(), _make_full_init())
    _write_cast_file(tmp_path, _CAST_FILE)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=60)

    assert not at.exception
    expander_labels = [e.label for e in at.expander]
    for instrument in ("Ed0", "EdZ", "LuZ", "EuZ"):
        assert f"{instrument} tilt" in expander_labels


def test_analyze_tab_ed0_stability_warns_when_outside_5_percent(tmp_path):
    ds = _make_minimal_dataset()
    # A brief, sharp mid-cast illumination dip -- Ed0's own (wide, depth.interval=10 m) smoothed
    # fit won't track it, so the raw/fitted correction ratio at those scans exceeds +/-5%.
    n = ds.sizes["time"]
    factor = np.ones(n)
    factor[n // 2 - 5 : n // 2 + 5] = 0.5
    ds["Ed0"] = ds["Ed0"] * factor[:, None]
    _write_nc(tmp_path, ds, _make_minimal_init())
    _write_cast_file(tmp_path, _CAST_FILE)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="analyze_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert any("outside the" in w.value and "+/-5%" in w.value for w in at.warning)
