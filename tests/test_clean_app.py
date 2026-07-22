from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from conftest import write_deployment  # noqa: E402

_APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "pycops" / "ui" / "clean_app.py")

CAST_1 = "WISE_CAST_001_190817_220856_URC.csv"
CAST_2 = "WISE_CAST_002_190817_221224_URC.csv"
CAST_3 = "WISE_CAST_003_190817_221636_URC.csv"


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


def test_app_missing_init_cops_dat_shows_error(tmp_path):
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert any("init.cops.dat" in e.value for e in at.error)


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
