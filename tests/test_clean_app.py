from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import pycops.io.discovery as discovery_module  # noqa: E402
import pycops.processing.deployment as deployment_module  # noqa: E402
from conftest import write_deployment  # noqa: E402
from pycops.io.discovery import DeploymentCastsResult  # noqa: E402
from pycops.processing.deployment import DeploymentProcessingResult  # noqa: E402
from pycops.processing.process_cast import process_cast  # noqa: E402

_APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "pycops" / "ui" / "clean_app.py")

CAST_1 = "WISE_CAST_001_190817_220856_URC.csv"
CAST_2 = "WISE_CAST_002_190817_221224_URC.csv"
CAST_3 = "WISE_CAST_003_190817_221636_URC.csv"


# -- Fixtures for the "Process casts" tab -----------------------------------------------------
# process_deployment()/read_deployment_casts() are monkeypatched in these tests rather than fed
# real raw cast files: producing a cast profile realistic enough (many scans over a real depth
# range) for the real fitting pipeline to succeed is already exercised by
# tests/test_deployment.py and tests/test_netcdf_deployment.py's own synthetic-dataset fixtures
# (reused here almost verbatim); these tests only need to verify the process tab's own
# orchestration -- directory checks, batch checkbox selection, progress/summary rendering, and
# per-deployment failure isolation -- not re-prove the numeric pipeline itself.

_PROC_WAVES = (340.0, 380.0, 443.0, 555.0)


def _make_processable_dataset(n=60, ed0_level=100.0):
    depth = np.linspace(0.05, 5.0, n)
    waves = np.array(_PROC_WAVES)
    K = np.array([2.0, 0.9, 0.3, 0.1])
    luz = np.array([0.0015, 0.006, 0.05, 0.3])[None, :] * np.exp(-K[None, :] * (depth + 0.238)[:, None])
    edz = np.array([50.0, 80.0, 150.0, 300.0])[None, :] * np.exp(-K[None, :] * (depth - 0.05)[:, None])
    ed0 = np.full((n, len(waves)), ed0_level)
    zeros = np.zeros(n)
    times = pd.date_range("2019-08-17T22:08:56", periods=n, freq="s")
    ds = xr.Dataset(
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
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.11626
    ds.attrs["latitude"] = 49.24872
    return ds


def _make_init_for_processing():
    nan = float("nan")
    return {
        "depth.is.on": "LuZ",
        "indice.water": 1.34,
        "rau.Fresnel": 0.043,
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 7.0, "LuZ": 7.0, "EuZ": 7.0},
        "radius.instrument.optics": {"Ed0": 0.035, "EdZ": 0.035, "LuZ": 0.035, "EuZ": 0.035},
        "delta.capteur.optics": {"Ed0": 0.0, "EdZ": -0.05, "LuZ": 0.238, "EuZ": 0.238},
        "sub.surface.removed.layer.optics": {"Ed0": 0.0, "EdZ": 0.3, "LuZ": 0.0, "EuZ": 0.0},
        "depth.interval.for.smoothing.optics": {"Ed0": 10.0, "EdZ": 3.0, "LuZ": 3.0, "EuZ": 3.0},
        "linear.fit.Rsquared.threshold.optics": {"Ed0": nan, "EdZ": 0.5, "LuZ": 0.5, "EuZ": 0.5},
        "linear.fit.max.delta.depth.optics": {"Ed0": nan, "EdZ": 3.0, "LuZ": 2.5, "EuZ": 2.5},
        "depth.discretization": [0, 0.01, 1, 0.02, 2, 0.05, 5, 0.1, 10, 0.2, 20, 0.5, 50, 1, 100, 2, 200, 5, 500],
    }


def _make_deployment_result_and_datasets():
    init = _make_init_for_processing()
    ds = _make_processable_dataset()
    return (
        DeploymentProcessingResult(
            cast_results={CAST_1: process_cast(ds, init)},
            bioshade_results={},
            bioshade_used=None,
            read_failures=[],
            processing_failures=[],
        ),
        {CAST_1: ds},
    )


def _patch_successful_processing(monkeypatch, only_for: str | None = None):
    """Makes discover_deployment/read_deployment_casts/process_deployment succeed with one
    synthetic cast for every deployment folder, or only for folders named ``only_for`` if given
    (used to simulate one broken deployment among several in a batch)."""
    result, datasets = _make_deployment_result_and_datasets()

    def fake_discover_deployment(directory):
        # deployment folders here are "<station>/cops" -- directory.name is always "cops", so
        # disambiguate on the station folder one level up instead.
        if only_for is not None and directory.parent.name != only_for:
            raise FileNotFoundError(f"simulated broken deployment: {directory}")
        return directory  # a real Deployment isn't needed -- read_deployment_casts is faked too

    monkeypatch.setattr(discovery_module, "discover_deployment", fake_discover_deployment)
    monkeypatch.setattr(
        discovery_module,
        "read_deployment_casts",
        lambda deployment: DeploymentCastsResult(datasets=datasets, failures=[]),
    )
    monkeypatch.setattr(deployment_module, "process_deployment", lambda directory: result)


def _write_l1_folder(tmp_path):
    from conftest import _write_cast_file

    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in (CAST_1, CAST_2):
        _write_cast_file(tmp_path, name)
    (tmp_path / "GPS_190817.tsv").write_text("dummy gps content\n")
    return tmp_path


def test_scaffold_tab_discovers_casts(tmp_path):
    l1 = _write_l1_folder(tmp_path / "L1")

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="scaffold_l1").set_value(str(l1)).run(timeout=30)

    assert not at.exception
    checkbox_labels = [c.label for c in at.checkbox if c.key and c.key.startswith("scaffold_cast_")]
    assert CAST_1 in checkbox_labels[0]
    assert CAST_2 in checkbox_labels[1]


def test_scaffold_tab_creates_station_and_copies_files(tmp_path):
    l1 = _write_l1_folder(tmp_path / "L1")
    l2_parent = tmp_path / "L2"

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="scaffold_l1").set_value(str(l1)).run(timeout=30)
    at.checkbox(key=f"scaffold_cast_{CAST_1}").set_value(True).run(timeout=30)
    at.checkbox(key=f"scaffold_cast_{CAST_2}").set_value(True).run(timeout=30)
    at.text_input(key="scaffold_station_id").set_value("BDA-01").run(timeout=30)
    at.text_input(key="scaffold_l2_parent").set_value(str(l2_parent)).run(timeout=30)
    at.button(key="scaffold_create").click().run(timeout=30)

    assert not at.exception
    assert any("Station created" in s.value for s in at.success)

    dest = l2_parent / "20190817_StationBDA-01" / "cops"
    assert (dest / CAST_1).exists()
    assert (dest / CAST_2).exists()
    assert (dest / "GPS_190817.tsv").exists()
    # L1 untouched
    assert (l1 / CAST_1).exists()


def test_scaffold_tab_only_checked_casts_are_included(tmp_path):
    l1 = _write_l1_folder(tmp_path / "L1")
    l2_parent = tmp_path / "L2"

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="scaffold_l1").set_value(str(l1)).run(timeout=30)
    at.checkbox(key=f"scaffold_cast_{CAST_1}").set_value(True).run(timeout=30)
    at.text_input(key="scaffold_station_id").set_value("BDA-01").run(timeout=30)
    at.text_input(key="scaffold_l2_parent").set_value(str(l2_parent)).run(timeout=30)
    at.button(key="scaffold_create").click().run(timeout=30)

    assert not at.exception
    dest = l2_parent / "20190817_StationBDA-01" / "cops"
    assert (dest / CAST_1).exists()
    assert not (dest / CAST_2).exists()


def test_scaffold_tab_generates_init_cops_dat_from_form(tmp_path):
    from pycops.io.config import read_init_cops

    l1 = _write_l1_folder(tmp_path / "L1")
    l2_parent = tmp_path / "L2"

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="scaffold_l1").set_value(str(l1)).run(timeout=30)
    at.checkbox(key=f"scaffold_cast_{CAST_1}").set_value(True).run(timeout=30)
    at.radio(key="scaffold_init_mode").set_value("Generate a new file").run(timeout=30)
    at.text_input(key="scaffold_station_id").set_value("BDA-01").run(timeout=30)
    at.text_input(key="scaffold_l2_parent").set_value(str(l2_parent)).run(timeout=30)
    at.button(key="scaffold_create").click().run(timeout=30)

    assert not at.exception
    dest = l2_parent / "20190817_StationBDA-01" / "cops"
    init_path = dest / "init.cops.dat"
    assert init_path.exists()

    params = read_init_cops(init_path)
    assert params["instruments.optics"] == ["Ed0", "EdZ", "LuZ"]
    assert params["depth.is.on"] == "LuZ"
    assert any("generated" in s.value for s in at.markdown)


def test_scaffold_tab_creating_station_jumps_to_clean_tab(tmp_path):
    """After creating a station, the researcher shouldn't have to go find the folder they just
    created in the cleaning tab -- it should jump there with the folder pre-filled."""
    from pycops.ui.clean_app import _TAB_CLEAN

    l1 = _write_l1_folder(tmp_path / "L1")
    l2_parent = tmp_path / "L2"

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="scaffold_l1").set_value(str(l1)).run(timeout=30)
    at.checkbox(key=f"scaffold_cast_{CAST_1}").set_value(True).run(timeout=30)
    at.checkbox(key=f"scaffold_cast_{CAST_2}").set_value(True).run(timeout=30)
    at.radio(key="scaffold_init_mode").set_value("Generate a new file").run(timeout=30)
    at.text_input(key="scaffold_station_id").set_value("BDA-01").run(timeout=30)
    at.text_input(key="scaffold_l2_parent").set_value(str(l2_parent)).run(timeout=30)
    at.button(key="scaffold_create").click().run(timeout=30)

    assert not at.exception
    dest = l2_parent / "20190817_StationBDA-01" / "cops"
    assert at.session_state["active_tab"] == _TAB_CLEAN
    assert at.text_input(key="clean_dir").value == str(dest)
    assert at.selectbox(key="clean_cast_select").options == [CAST_1, CAST_2]


def test_app_discovers_casts_and_renders_slider(tmp_path):
    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert not at.error
    assert at.selectbox(key="clean_cast_select").options == [
        "WISE_CAST_001_190817_220856_URC.csv",
        "WISE_CAST_002_190817_221224_URC.csv",
        "WISE_CAST_003_190817_221636_URC.csv",
    ]
    assert len(at.slider) == 1
    # select.cops.dat's fixture row for cast 001: flag=1, method=Rrs.0p, extra=NA
    assert at.selectbox(key=f"clean_flag::{CAST_1}").value == 1
    assert at.selectbox(key=f"clean_method::{CAST_1}").value == "Rrs.0p"
    assert at.checkbox(key=f"clean_shallow::{CAST_1}").value is False


def test_app_save_advances_to_next_cast_with_its_own_values(tmp_path):
    """Regression test: 'Save && next' previously appeared to do nothing, because the selectbox's
    own sticky widget state clobbered the programmatic cast_idx advance, and every per-cast field
    shared one fixed key so it kept showing whatever was last typed instead of the next cast's own
    saved info.cops.dat values."""
    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)
    assert at.selectbox(key="clean_cast_select").value == CAST_1

    # Edit cast 1's longitude but don't save it -- shouldn't leak into cast 2's field.
    at.text_input(key=f"clean_lon::{CAST_1}").set_value("-1.111").run(timeout=30)
    at.button(key="clean_save").click().run(timeout=30)

    assert not at.exception
    assert at.selectbox(key="clean_cast_select").value == CAST_2
    assert at.text_input(key=f"clean_lon::{CAST_2}").value == "-68.11626"


def test_clean_tab_chl_mode_defaults_from_existing_flag(tmp_path):
    """Fixture rows: cast 001 has chl=0, cast 002 has chl=NA (none), cast 003 has chl=0.2 (Morel &
    Maritorena). The dropdown must faithfully reflect each existing row's actual value (Simon:
    "si une station existe déjà, l'interface doit refléter le contenu du fichier") -- cast 001's
    chl=0 defaults to "From a file", not "Kd-derived". Kd-derived (999) is only the default for a
    cast with no info.cops.dat row at all yet, see
    test_clean_tab_new_cast_without_info_row_defaults_to_kd_derived below."""
    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)

    assert at.selectbox(key=f"clean_chl_mode::{CAST_1}").value == "From a file (absorption.cops.dat)"

    at.selectbox(key="clean_cast_select").set_value(CAST_2).run(timeout=30)
    assert at.selectbox(key=f"clean_chl_mode::{CAST_2}").value == "No shadow correction"

    at.selectbox(key="clean_cast_select").set_value(CAST_3).run(timeout=30)
    assert at.selectbox(key=f"clean_chl_mode::{CAST_3}").value == "Morel & Maritorena (chlorophyll)"
    assert at.text_input(key=f"clean_chl_conc::{CAST_3}").value == "0.2"


def test_clean_tab_chl_mode_from_file_round_trips_without_mutating_it(tmp_path):
    """Regression test: saving a cast for an unrelated reason (e.g. just moving the time-window
    slider) must not silently rewrite an existing chl=0 row to chl=999 -- the Save button always
    resends the currently-displayed mode, so the displayed mode must match the file first."""
    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)
    at.slider(key=f"clean_time_window::{CAST_1}").set_value((0.0, 0.05)).run(timeout=30)
    at.button(key="clean_save").click().run(timeout=30)

    assert not at.exception
    info_text = (tmp_path / "info.cops.dat").read_text()
    assert "WISE_CAST_001_190817_220856_URC.csv;-68.11626;49.24872;0;0,0.05;" in info_text


def test_clean_tab_new_cast_without_info_row_defaults_to_kd_derived(tmp_path):
    """A cast with no info.cops.dat row at all yet (brand-new station) defaults to Kd-derived
    (chl=999), per Simon: new stations should default to Kd-derived rather than an unset value."""
    write_deployment(tmp_path)
    (tmp_path / "info.cops.dat").unlink()

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)

    assert at.selectbox(key=f"clean_chl_mode::{CAST_1}").value == "Kd-derived (default)"

    at.button(key="clean_save").click().run(timeout=30)
    assert not at.exception
    info_text = (tmp_path / "info.cops.dat").read_text()
    row = next(line for line in info_text.splitlines() if line.startswith(CAST_1))
    assert row.split(";")[3] == "999"


def test_clean_tab_morel_maritorena_concentration_round_trips(tmp_path):
    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)
    at.selectbox(key="clean_cast_select").set_value(CAST_2).run(timeout=30)
    at.selectbox(key=f"clean_chl_mode::{CAST_2}").set_value("Morel & Maritorena (chlorophyll)").run(timeout=30)
    assert at.text_input(key=f"clean_chl_conc::{CAST_2}").value == "1"

    at.text_input(key=f"clean_chl_conc::{CAST_2}").set_value("2.5").run(timeout=30)
    at.button(key="clean_save").click().run(timeout=30)

    assert not at.exception
    info_text = (tmp_path / "info.cops.dat").read_text()
    assert "WISE_CAST_002_190817_221224_URC.csv;-68.11626;49.24872;2.5;" in info_text


def test_clean_tab_shows_gps_hint_when_position_missing(tmp_path):
    write_deployment(tmp_path)
    (tmp_path / "GPS_190817.tsv").write_text("dummy gps content\n")

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)
    at.selectbox(key="clean_cast_select").set_value(CAST_2).run(timeout=30)

    # cast 002 already has a real lon/lat in the fixture -- no hint yet.
    assert not any("GPS" in m.value for m in at.markdown)

    at.text_input(key=f"clean_lon::{CAST_2}").set_value("NA").run(timeout=30)
    at.text_input(key=f"clean_lat::{CAST_2}").set_value("NA").run(timeout=30)

    # rendered as st.markdown (not st.caption) so it can be styled larger/red -- easy to miss
    # otherwise, since a missing position silently disables shadow correction.
    assert any("GPS_190817.tsv" in m.value for m in at.markdown)


def test_clean_tab_shows_unsaved_position_hint(tmp_path):
    """Regression test: typing a new position clears the top warning immediately (it reacts to
    the live text field), but nothing is persisted until Save is clicked -- without this caption,
    a researcher could easily believe the position is already handled."""
    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not any("not saved yet" in c.value for c in at.caption)

    at.text_input(key=f"clean_lon::{CAST_1}").set_value("-1.234").run(timeout=30)

    assert any("not saved yet" in c.value for c in at.caption)

    at.button(key="clean_save").click().run(timeout=30)

    assert not at.exception
    assert not any("not saved yet" in c.value for c in at.caption)


def test_clean_tab_prepare_another_station_jumps_to_scaffold_tab(tmp_path):
    from pycops.ui.clean_app import _TAB_SCAFFOLD

    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)
    at.selectbox(key="clean_cast_select").set_value(CAST_3).run(timeout=30)
    at.button(key="clean_save").click().run(timeout=30)

    assert not at.exception
    assert any(b.key == "clean_prepare_another" for b in at.button)

    at.button(key="clean_prepare_another").click().run(timeout=30)

    assert not at.exception
    assert at.session_state["active_tab"] == _TAB_SCAFFOLD


def test_app_missing_init_cops_dat_shows_error(tmp_path):
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert any("init.cops.dat" in e.value for e in at.error)


def test_clean_tab_init_cops_dat_editor_prefills_and_saves(tmp_path):
    from pycops.io.config import read_init_cops

    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)

    # conftest's INIT_COPS_DAT fixture has no windspeed_ms -- read_init_cops backfills 4.0.
    assert at.number_input(key="clean_init_windspeed").value == 4.0
    at.number_input(key="clean_init_windspeed").set_value(6.5).run(timeout=30)
    at.button(key="clean_init_save").click().run(timeout=30)

    assert not at.exception
    assert any("Saved init.cops.dat" in t.value for t in at.toast)
    assert read_init_cops(tmp_path / "init.cops.dat")["windspeed_ms"] == 6.5


def test_clean_tab_next_to_process_appears_only_once_all_casts_cleaned(tmp_path):
    """Fixture: select.cops.dat only covers casts 1/2, so cast 3 starts out not-yet-cleaned --
    the "Next -> Process casts" button shouldn't appear until it is."""
    from pycops.ui.clean_app import _TAB_PROCESS

    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not any(b.key == "clean_next_to_process" for b in at.button)

    at.selectbox(key="clean_cast_select").set_value(CAST_3).run(timeout=30)
    at.button(key="clean_save").click().run(timeout=30)

    assert not at.exception
    assert any(b.key == "clean_next_to_process" for b in at.button)

    at.button(key="clean_next_to_process").click().run(timeout=30)

    assert not at.exception
    assert at.session_state["active_tab"] == _TAB_PROCESS
    assert at.text_input(key="process_dir").value == str(tmp_path)


def test_clean_tab_next_to_process_blocked_when_position_unresolved(tmp_path):
    """A cast with no info.cops.dat position and no GPS file in the folder must block
    'Next -> Process casts' even once every cast has an info.cops.dat/select.cops.dat row --
    Simon: "on ne doit pas être en mesure de passer au processing sans une position lat/lon"."""
    from pycops.io.config import update_cast_info

    write_deployment(tmp_path)
    update_cast_info(tmp_path / "info.cops.dat", CAST_3, longitude=None, latitude=None)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)
    at.selectbox(key="clean_cast_select").set_value(CAST_3).run(timeout=30)
    at.button(key="clean_save").click().run(timeout=30)

    assert not at.exception
    assert not any(b.key == "clean_next_to_process" for b in at.button)
    assert any("position unresolved" in e.value and CAST_3 in e.value for e in at.error)


def test_app_save_writes_time_window(tmp_path):
    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)
    at.slider(key=f"clean_time_window::{CAST_1}").set_value((0.0, 0.05)).run(timeout=30)
    at.button(key="clean_save").click().run(timeout=30)

    assert not at.exception
    assert any("Saved info.cops.dat" in t.value for t in at.toast)

    info_text = (tmp_path / "info.cops.dat").read_text()
    # position/chl round-trip unchanged (full precision preserved), time.window updated; the
    # Save button always resends every field, so untouched-but-blank fields canonicalize to "x".
    assert "WISE_CAST_001_190817_220856_URC.csv;-68.11626;49.24872;0;0,0.05;x;x;x;x;x" in info_text


def test_app_save_writes_select_cops_dat(tmp_path):
    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)
    at.selectbox(key=f"clean_flag::{CAST_1}").set_value(0).run(timeout=30)  # QC flag -> Reject
    at.selectbox(key=f"clean_method::{CAST_1}").set_value("Rrs.0p.linear").run(timeout=30)
    at.checkbox(key=f"clean_shallow::{CAST_1}").set_value(True).run(timeout=30)
    at.slider(key=f"clean_time_window::{CAST_1}").set_value((0.0, 0.05)).run(timeout=30)
    at.button(key="clean_save").click().run(timeout=30)

    assert not at.exception
    assert any("select.cops.dat" in t.value for t in at.toast)

    select_text = (tmp_path / "select.cops.dat").read_text()
    assert "WISE_CAST_001_190817_220856_URC.csv;0;Rrs.0p.linear;1" in select_text


def test_directory_browser_navigates_and_selects_without_crashing(tmp_path):
    # A native tkinter dialog crashed the whole app on macOS (Streamlit runs script logic on a
    # worker thread; Tk/AppKit windows must be created on the main thread) -- this in-app
    # Streamlit-only browser replaced it; confirm descending, ascending, and picking all work.
    (tmp_path / "sub_a").mkdir()
    (tmp_path / "sub_a" / "sub_b").mkdir()

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="scaffold_l1").set_value(str(tmp_path)).run(timeout=30)

    at.button(key="scaffold_l1_sub_sub_a").click().run(timeout=30)
    assert not at.exception
    assert any(str(tmp_path / "sub_a") in c.value for c in at.caption)

    at.button(key="scaffold_l1_sub_sub_b").click().run(timeout=30)
    assert not at.exception
    assert any(str(tmp_path / "sub_a" / "sub_b") in c.value for c in at.caption)

    at.button(key="scaffold_l1_up").click().run(timeout=30)
    assert not at.exception
    assert any(str(tmp_path / "sub_a") in c.value for c in at.caption)

    at.button(key="scaffold_l1_choose").click().run(timeout=30)
    assert not at.exception
    assert at.text_input(key="scaffold_l1").value == str(tmp_path / "sub_a")


def test_directory_browsers_default_to_project_root_instead_of_home(tmp_path):
    """Simon: landing in $HOME every time a Browse popover opens was annoying since every real
    project folder lives elsewhere -- setting the project-root folder once (tab 1) should become
    every other, still-empty directory input's own Browse default, not just its own."""
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="project_root_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    # scaffold_l2_parent (tab 1) and process_parent (tab 3) are both still empty -- their own
    # Browse popovers must fall back to the just-set project root, not str(Path.home()).
    matching = [c for c in at.caption if str(tmp_path) in c.value]
    assert len(matching) >= 2


def test_process_tab_single_deployment_writes_nc_files(tmp_path, monkeypatch):
    _patch_successful_processing(monkeypatch)
    (tmp_path / "init.cops.dat").write_text("")

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="process_dir").set_value(str(tmp_path)).run(timeout=30)
    at.button(key="process_single_run").click().run(timeout=30)

    assert not at.exception
    assert (tmp_path / "nc" / f"{Path(CAST_1).stem}.nc").exists()
    assert any("1 cast(s) processed" in m.value for m in at.markdown)


def test_process_tab_analyze_button_jumps_to_analyze_tab_with_folder_prefilled(tmp_path, monkeypatch):
    from pycops.ui.clean_app import _TAB_ANALYZE

    _patch_successful_processing(monkeypatch)
    (tmp_path / "init.cops.dat").write_text("")

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="process_dir").set_value(str(tmp_path)).run(timeout=30)
    at.button(key="process_to_analyze").click().run(timeout=30)

    assert not at.exception
    assert at.session_state["active_tab"] == _TAB_ANALYZE
    assert at.text_input(key="analyze_dir").value == str(tmp_path)


def test_process_tab_missing_init_cops_dat_shows_error(tmp_path):
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="process_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert any("init.cops.dat" in e.value for e in at.error)


def test_process_tab_batch_discovers_and_processes_multiple_deployments(tmp_path, monkeypatch):
    _patch_successful_processing(monkeypatch)
    parent = tmp_path / "L2"
    (parent / "StationA" / "cops").mkdir(parents=True)
    (parent / "StationA" / "cops" / "init.cops.dat").write_text("")
    (parent / "StationB" / "cops").mkdir(parents=True)
    (parent / "StationB" / "cops" / "init.cops.dat").write_text("")

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.radio(key="process_mode").set_value("Batch (multiple deployments)").run(timeout=30)
    at.text_input(key="process_parent").set_value(str(parent)).run(timeout=30)

    assert not at.exception
    assert at.checkbox(key=f"process_batch_{Path('StationA') / 'cops'}").value is True
    assert at.checkbox(key=f"process_batch_{Path('StationB') / 'cops'}").value is True

    at.button(key="process_batch_run").click().run(timeout=30)

    assert not at.exception
    assert (parent / "StationA" / "cops" / "nc" / f"{Path(CAST_1).stem}.nc").exists()
    assert (parent / "StationB" / "cops" / "nc" / f"{Path(CAST_1).stem}.nc").exists()
    assert any("2 ok, 0 with warnings, 0 failed" in m.value for m in at.markdown)


def test_process_tab_batch_unchecking_excludes_it(tmp_path, monkeypatch):
    _patch_successful_processing(monkeypatch)
    parent = tmp_path / "L2"
    (parent / "StationA" / "cops").mkdir(parents=True)
    (parent / "StationA" / "cops" / "init.cops.dat").write_text("")
    (parent / "StationB" / "cops").mkdir(parents=True)
    (parent / "StationB" / "cops" / "init.cops.dat").write_text("")

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.radio(key="process_mode").set_value("Batch (multiple deployments)").run(timeout=30)
    at.text_input(key="process_parent").set_value(str(parent)).run(timeout=30)
    at.checkbox(key=f"process_batch_{Path('StationB') / 'cops'}").set_value(False).run(timeout=30)
    at.button(key="process_batch_run").click().run(timeout=30)

    assert not at.exception
    assert (parent / "StationA" / "cops" / "nc" / f"{Path(CAST_1).stem}.nc").exists()
    assert not (parent / "StationB" / "cops" / "nc").exists()


def test_process_tab_batch_isolates_a_broken_deployment(tmp_path, monkeypatch):
    """One deployment (StationBad) fails during discovery (e.g. a missing/malformed config) --
    the batch must still finish and still process the good one, reporting the bad one's error
    rather than aborting."""
    _patch_successful_processing(monkeypatch, only_for="StationGood")
    parent = tmp_path / "L2"
    (parent / "StationGood" / "cops").mkdir(parents=True)
    (parent / "StationGood" / "cops" / "init.cops.dat").write_text("")
    (parent / "StationBad" / "cops").mkdir(parents=True)
    (parent / "StationBad" / "cops" / "init.cops.dat").write_text("")

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.radio(key="process_mode").set_value("Batch (multiple deployments)").run(timeout=30)
    at.text_input(key="process_parent").set_value(str(parent)).run(timeout=30)
    at.button(key="process_batch_run").click().run(timeout=30)

    assert not at.exception
    assert (parent / "StationGood" / "cops" / "nc" / f"{Path(CAST_1).stem}.nc").exists()
    assert not (parent / "StationBad" / "cops" / "nc").exists()
    assert any("simulated broken deployment" in e.value for e in at.error)
    assert any("1 ok, 0 with warnings, 1 failed" in m.value for m in at.markdown)
