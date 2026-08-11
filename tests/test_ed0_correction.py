from __future__ import annotations

from pycops.io.ed0_correction import read_ed0_correction_methods, update_ed0_correction_method


def test_read_ed0_correction_methods_missing_file_returns_empty_dict(tmp_path):
    assert read_ed0_correction_methods(tmp_path / "ed0_correction_method.cops.dat") == {}


def test_update_then_read_round_trips(tmp_path):
    path = tmp_path / "ed0_correction_method.cops.dat"
    update_ed0_correction_method(path, "CAST_001.tsv", "smoothed")

    assert read_ed0_correction_methods(path) == {"CAST_001.tsv": "smoothed"}


def test_update_only_touches_the_named_cast_row(tmp_path):
    path = tmp_path / "ed0_correction_method.cops.dat"
    update_ed0_correction_method(path, "CAST_001.tsv", "smoothed")
    update_ed0_correction_method(path, "CAST_002.tsv", "raw")
    update_ed0_correction_method(path, "CAST_001.tsv", "raw")

    methods = read_ed0_correction_methods(path)
    assert methods == {"CAST_001.tsv": "raw", "CAST_002.tsv": "raw"}


def test_update_with_none_removes_the_row(tmp_path):
    path = tmp_path / "ed0_correction_method.cops.dat"
    update_ed0_correction_method(path, "CAST_001.tsv", "smoothed")
    update_ed0_correction_method(path, "CAST_002.tsv", "raw")
    update_ed0_correction_method(path, "CAST_001.tsv", None)

    assert read_ed0_correction_methods(path) == {"CAST_002.tsv": "raw"}


def test_update_preserves_other_rows_byte_for_byte(tmp_path):
    path = tmp_path / "ed0_correction_method.cops.dat"
    path.write_text("CAST_001.tsv;raw\r\nCAST_002.tsv;smoothed\r\n", newline="")

    update_ed0_correction_method(path, "CAST_001.tsv", "smoothed")

    with path.open(newline="") as f:
        text = f.read()
    assert text == "CAST_001.tsv;smoothed\r\nCAST_002.tsv;smoothed\r\n"


def test_update_on_nonexistent_file_with_none_does_not_create_it(tmp_path):
    path = tmp_path / "ed0_correction_method.cops.dat"
    update_ed0_correction_method(path, "CAST_001.tsv", None)

    assert not path.exists()
