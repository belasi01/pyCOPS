from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from conftest import write_deployment  # noqa: E402

_APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "pycops" / "ui" / "clean_app.py")


def test_app_discovers_casts_and_renders_slider(tmp_path):
    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input[0].set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert not at.error
    assert at.selectbox[0].options == [
        "WISE_CAST_001_190817_220856_URC.csv",
        "WISE_CAST_002_190817_221224_URC.csv",
        "WISE_CAST_003_190817_221636_URC.csv",
    ]
    assert len(at.slider) == 1
    # select.cops.dat's fixture row for cast 001: flag=1, method=Rrs.0p, extra=NA
    assert at.selectbox[1].value == 1
    assert at.selectbox[2].value == "Rrs.0p"
    assert at.checkbox[0].value is False


def test_app_missing_init_cops_dat_shows_error(tmp_path):
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input[0].set_value(str(tmp_path)).run(timeout=30)

    assert not at.exception
    assert any("init.cops.dat" in e.value for e in at.error)


def test_app_save_writes_time_window(tmp_path):
    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input[0].set_value(str(tmp_path)).run(timeout=30)
    at.slider[0].set_value((0.0, 0.05)).run(timeout=30)
    at.button[0].click().run(timeout=30)

    assert not at.exception
    assert any("Saved time.window" in s.value for s in at.success)

    info_text = (tmp_path / "info.cops.dat").read_text()
    assert "WISE_CAST_001_190817_220856_URC.csv;-68.11626;49.24872;0;0,0.05" in info_text


def test_app_save_writes_select_cops_dat(tmp_path):
    write_deployment(tmp_path)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input[0].set_value(str(tmp_path)).run(timeout=30)
    at.selectbox[1].set_value(0).run(timeout=30)  # QC flag -> Reject
    at.selectbox[2].set_value("Rrs.0p.linear").run(timeout=30)  # Rrs method
    at.checkbox[0].set_value(True).run(timeout=30)  # Shallow
    at.slider[0].set_value((0.0, 0.05)).run(timeout=30)
    at.button[0].click().run(timeout=30)

    assert not at.exception
    assert any("select.cops.dat" in s.value for s in at.success)

    select_text = (tmp_path / "select.cops.dat").read_text()
    assert "WISE_CAST_001_190817_220856_URC.csv;0;Rrs.0p.linear;1" in select_text
