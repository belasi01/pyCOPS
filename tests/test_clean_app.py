from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from conftest import write_deployment  # noqa: E402

_APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "pycops" / "ui" / "clean_app.py")

CAST_1 = "WISE_CAST_001_190817_220856_URC.csv"
CAST_2 = "WISE_CAST_002_190817_221224_URC.csv"


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
    at.text_input(key="scaffold_station_id").set_value("BDA-01").run(timeout=30)
    at.text_input(key="scaffold_l2_parent").set_value(str(l2_parent)).run(timeout=30)
    at.button(key="scaffold_create").click().run(timeout=30)

    assert not at.exception
    assert any("Station créée" in s.value for s in at.success)

    dest = l2_parent / "20190817_StationBDA-01" / "cops"
    assert (dest / CAST_1).exists()
    assert (dest / CAST_2).exists()
    assert (dest / "GPS_190817.tsv").exists()
    # L1 untouched
    assert (l1 / CAST_1).exists()


def test_scaffold_tab_unchecking_a_cast_excludes_it(tmp_path):
    l1 = _write_l1_folder(tmp_path / "L1")
    l2_parent = tmp_path / "L2"

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="scaffold_l1").set_value(str(l1)).run(timeout=30)
    at.checkbox(key=f"scaffold_cast_{CAST_2}").set_value(False).run(timeout=30)
    at.text_input(key="scaffold_station_id").set_value("BDA-01").run(timeout=30)
    at.text_input(key="scaffold_l2_parent").set_value(str(l2_parent)).run(timeout=30)
    at.button(key="scaffold_create").click().run(timeout=30)

    assert not at.exception
    dest = l2_parent / "20190817_StationBDA-01" / "cops"
    assert (dest / CAST_1).exists()
    assert not (dest / CAST_2).exists()


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
    assert at.selectbox(key="clean_flag").value == 1
    assert at.selectbox(key="clean_method").value == "Rrs.0p"
    assert at.checkbox(key="clean_shallow").value is False


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
    at.slider(key="clean_time_window").set_value((0.0, 0.05)).run(timeout=30)
    at.button(key="clean_save").click().run(timeout=30)

    assert not at.exception
    assert any("Saved info.cops.dat" in s.value for s in at.success)

    info_text = (tmp_path / "info.cops.dat").read_text()
    # position/chl round-trip unchanged (full precision preserved), time.window updated; the
    # Save button always resends every field, so untouched-but-blank fields canonicalize to "x".
    assert "WISE_CAST_001_190817_220856_URC.csv;-68.11626;49.24872;0;0,0.05;x;x;x;x;x" in info_text


def test_app_save_writes_select_cops_dat(tmp_path):
    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input(key="clean_dir").set_value(str(tmp_path)).run(timeout=30)
    at.selectbox(key="clean_flag").set_value(0).run(timeout=30)  # QC flag -> Reject
    at.selectbox(key="clean_method").set_value("Rrs.0p.linear").run(timeout=30)
    at.checkbox(key="clean_shallow").set_value(True).run(timeout=30)
    at.slider(key="clean_time_window").set_value((0.0, 0.05)).run(timeout=30)
    at.button(key="clean_save").click().run(timeout=30)

    assert not at.exception
    assert any("select.cops.dat" in s.value for s in at.success)

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
