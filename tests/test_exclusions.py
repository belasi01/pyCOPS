from __future__ import annotations

from pycops.io.exclusions import read_wavelength_exclusions, update_wavelength_exclusions


def test_read_wavelength_exclusions_missing_file_returns_empty_dict(tmp_path):
    assert read_wavelength_exclusions(tmp_path / "rrs_wavelength_exclusions.cops.dat") == {}


def test_update_then_read_round_trips(tmp_path):
    path = tmp_path / "rrs_wavelength_exclusions.cops.dat"
    update_wavelength_exclusions(path, "CAST_001.tsv", [380.0, 410.0])

    assert read_wavelength_exclusions(path) == {"CAST_001.tsv": [380.0, 410.0]}


def test_update_only_touches_the_named_cast_row(tmp_path):
    path = tmp_path / "rrs_wavelength_exclusions.cops.dat"
    update_wavelength_exclusions(path, "CAST_001.tsv", [380.0])
    update_wavelength_exclusions(path, "CAST_002.tsv", [443.0, 555.0])
    update_wavelength_exclusions(path, "CAST_001.tsv", [340.0, 380.0])

    exclusions = read_wavelength_exclusions(path)
    assert exclusions == {
        "CAST_001.tsv": [340.0, 380.0],
        "CAST_002.tsv": [443.0, 555.0],
    }


def test_update_with_empty_list_removes_the_row(tmp_path):
    path = tmp_path / "rrs_wavelength_exclusions.cops.dat"
    update_wavelength_exclusions(path, "CAST_001.tsv", [380.0])
    update_wavelength_exclusions(path, "CAST_002.tsv", [443.0])
    update_wavelength_exclusions(path, "CAST_001.tsv", [])

    assert read_wavelength_exclusions(path) == {"CAST_002.tsv": [443.0]}


def test_update_preserves_other_rows_byte_for_byte(tmp_path):
    path = tmp_path / "rrs_wavelength_exclusions.cops.dat"
    path.write_text("CAST_001.tsv;380\r\nCAST_002.tsv;443,555\r\n", newline="")

    update_wavelength_exclusions(path, "CAST_001.tsv", [340.0, 380.0])

    with path.open(newline="") as f:
        text = f.read()
    assert text == "CAST_001.tsv;340,380\r\nCAST_002.tsv;443,555\r\n"


def test_update_on_nonexistent_file_with_empty_list_does_not_create_it(tmp_path):
    path = tmp_path / "rrs_wavelength_exclusions.cops.dat"
    update_wavelength_exclusions(path, "CAST_001.tsv", [])

    assert not path.exists()
