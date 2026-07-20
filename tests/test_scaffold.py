from __future__ import annotations

from datetime import date

import pytest

from pycops.io.scaffold import (
    discover_l1_casts,
    find_gps_files_for_date,
    scaffold_station,
    validate_station_id,
)

CAST_1 = "WISE_CAST_001_190817_220856_URC.csv"
CAST_2 = "WISE_CAST_002_190817_221224_URC.csv"
CAST_OTHER_DAY = "WISE_CAST_001_190818_090000_URC.csv"
GPS_FILE = "GPS_190817.tsv"


def _write_l1_folder(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in (CAST_1, CAST_2, CAST_OTHER_DAY):
        (tmp_path / name).write_text("dummy cast content\n")
    (tmp_path / GPS_FILE).write_text("dummy gps content\n")
    (tmp_path / f"{CAST_1[:-8]}_LOG.csv").write_text("dummy log, not a cast\n")
    return tmp_path


def test_discover_l1_casts_finds_urc_files_sorted_by_date(tmp_path):
    _write_l1_folder(tmp_path)

    casts = discover_l1_casts(tmp_path)

    assert [p.name for p in casts] == [CAST_1, CAST_2, CAST_OTHER_DAY]


def test_discover_l1_casts_skips_non_urc_siblings(tmp_path):
    _write_l1_folder(tmp_path)

    casts = discover_l1_casts(tmp_path)

    assert all("_LOG" not in p.name for p in casts)


def test_find_gps_files_for_date(tmp_path):
    _write_l1_folder(tmp_path)

    gps = find_gps_files_for_date(tmp_path, date(2019, 8, 17))

    assert [p.name for p in gps] == [GPS_FILE]


def test_find_gps_files_for_date_no_match(tmp_path):
    _write_l1_folder(tmp_path)

    gps = find_gps_files_for_date(tmp_path, date(2020, 1, 1))

    assert gps == []


def test_validate_station_id_strips_whitespace():
    assert validate_station_id("  MAN-F05  ") == "MAN-F05"


def test_validate_station_id_rejects_blank():
    with pytest.raises(ValueError, match="blank"):
        validate_station_id("   ")


@pytest.mark.parametrize("bad_id", ["a/b", "a\\b", ".", ".."])
def test_validate_station_id_rejects_path_like(bad_id):
    with pytest.raises(ValueError, match="path"):
        validate_station_id(bad_id)


def test_scaffold_station_creates_expected_folder_and_copies_files(tmp_path):
    l1 = _write_l1_folder(tmp_path / "L1")
    l2_parent = tmp_path / "L2"

    result = scaffold_station(l1, l2_parent, "MAN-F05", [CAST_1, CAST_2])

    assert result.destination == l2_parent / "20190817_StationMAN-F05" / "cops"
    assert result.destination.is_dir()
    assert sorted(result.copied_casts) == [CAST_1, CAST_2]
    assert result.copied_gps_files == [GPS_FILE]
    assert (result.destination / CAST_1).exists()
    assert (result.destination / CAST_2).exists()
    assert (result.destination / GPS_FILE).exists()
    assert result.date_used == date(2019, 8, 17)
    # L1 is never touched
    assert (l1 / CAST_1).exists()
    assert (l1 / CAST_2).exists()


def test_scaffold_station_warns_on_mismatched_dates(tmp_path):
    l1 = _write_l1_folder(tmp_path / "L1")
    l2_parent = tmp_path / "L2"

    with pytest.warns(UserWarning, match="multiple dates"):
        result = scaffold_station(l1, l2_parent, "MAN-F05", [CAST_1, CAST_OTHER_DAY])

    # still uses the first selected cast's date
    assert result.date_used == date(2019, 8, 17)
    assert sorted(result.copied_casts) == sorted([CAST_1, CAST_OTHER_DAY])


def test_scaffold_station_does_not_overwrite_by_default(tmp_path):
    l1 = _write_l1_folder(tmp_path / "L1")
    l2_parent = tmp_path / "L2"

    first = scaffold_station(l1, l2_parent, "MAN-F05", [CAST_1])
    (first.destination / CAST_1).write_text("locally edited, must not be clobbered\n")

    second = scaffold_station(l1, l2_parent, "MAN-F05", [CAST_1])

    assert second.copied_casts == []
    assert set(second.skipped_existing) == {CAST_1, GPS_FILE}  # GPS also already copied by `first`
    assert (second.destination / CAST_1).read_text() == "locally edited, must not be clobbered\n"


def test_scaffold_station_overwrite_true_replaces_existing(tmp_path):
    l1 = _write_l1_folder(tmp_path / "L1")
    l2_parent = tmp_path / "L2"

    first = scaffold_station(l1, l2_parent, "MAN-F05", [CAST_1])
    (first.destination / CAST_1).write_text("stale copy\n")

    second = scaffold_station(l1, l2_parent, "MAN-F05", [CAST_1], overwrite=True)

    assert second.copied_casts == [CAST_1]
    assert (second.destination / CAST_1).read_text() == "dummy cast content\n"


def test_scaffold_station_copies_init_cops_dat_template(tmp_path):
    l1 = _write_l1_folder(tmp_path / "L1")
    l2_parent = tmp_path / "L2"
    template = tmp_path / "template_init.cops.dat"
    template.write_text("verbose;logical;TRUE\n")

    result = scaffold_station(l1, l2_parent, "MAN-F05", [CAST_1], init_cops_dat_template=template)

    assert result.copied_init is True
    assert (result.destination / "init.cops.dat").read_text() == "verbose;logical;TRUE\n"


def test_scaffold_station_raises_on_no_casts_selected(tmp_path):
    l1 = _write_l1_folder(tmp_path / "L1")
    with pytest.raises(ValueError, match="no cast files selected"):
        scaffold_station(l1, tmp_path / "L2", "MAN-F05", [])
