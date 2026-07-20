from __future__ import annotations

from conftest import write_deployment, write_deployment_with_bad_cast
from pycops.io.discovery import (
    CastSelection,
    discover_deployment,
    read_deployment_casts,
    read_select_cops,
    update_cast_selection,
)


def test_cast_selection_shallow_true_for_flag_field():
    assert CastSelection(file="x", flag=1, method="Rrs.0p", extra="1").shallow is True


def test_cast_selection_shallow_false_for_na_or_other():
    assert CastSelection(file="x", flag=1, method="Rrs.0p", extra="NA").shallow is False
    assert CastSelection(file="x", flag=1, method="Rrs.0p", extra="0").shallow is False
    assert CastSelection(file="x", flag=1, method="Rrs.0p", extra="").shallow is False


def test_update_cast_selection_creates_missing_file(tmp_path):
    path = tmp_path / "select.cops.dat"
    update_cast_selection(path, "WISE_CAST_001_190817_220856_URC.csv", 1, "Rrs.0p", shallow=True)

    entries = read_select_cops(path)
    assert len(entries) == 1
    assert entries[0].flag == 1
    assert entries[0].method == "Rrs.0p"
    assert entries[0].shallow is True


def test_update_cast_selection_replaces_existing_row_preserving_others(tmp_path):
    write_deployment(tmp_path)
    path = tmp_path / "select.cops.dat"
    before = path.read_text().splitlines()

    update_cast_selection(path, "WISE_CAST_001_190817_220856_URC.csv", 0, "Rrs.0p.linear", shallow=False)

    after = path.read_text().splitlines()
    changed = [i for i in range(len(before)) if before[i] != after[i]]
    assert len(changed) == 1

    entries = read_select_cops(path)
    updated = next(e for e in entries if e.file == "WISE_CAST_001_190817_220856_URC.csv")
    assert updated.flag == 0
    assert updated.method == "Rrs.0p.linear"
    assert updated.shallow is False
    # the other row (cast 002, untouched by write_deployment's SELECT_COPS_DAT) stays as-is
    other = next(e for e in entries if e.file == "WISE_CAST_002_190817_221224_URC.csv")
    assert other.flag == 0
    assert other.method == "Rrs.0p.linear"


def test_update_cast_selection_appends_row_when_cast_not_listed(tmp_path):
    write_deployment(tmp_path)
    path = tmp_path / "select.cops.dat"

    update_cast_selection(path, "WISE_CAST_003_190817_221636_URC.csv", 1, "Rrs.0p", shallow=True)

    entries = read_select_cops(path)
    assert len(entries) == 3
    new_entry = next(e for e in entries if e.file == "WISE_CAST_003_190817_221636_URC.csv")
    assert new_entry.flag == 1
    assert new_entry.shallow is True


def test_update_cast_selection_preserves_crlf(tmp_path):
    path = tmp_path / "select.cops.dat"
    with path.open("w", newline="") as f:
        f.write("WISE_CAST_001_190817_220856_URC.csv;1;Rrs.0p;NA\r\nWISE_CAST_002_x_y_URC.csv;1;Rrs.0p;NA\r\n")

    update_cast_selection(path, "WISE_CAST_001_190817_220856_URC.csv", 0, "Rrs.0p.linear", shallow=False)

    raw = path.read_bytes()
    assert raw.count(b"\r\n") == 2
    assert b"\n\n" not in raw.replace(b"\r\n", b"")


def test_discover_deployment_reads_all_casts(tmp_path):
    write_deployment(tmp_path)
    deployment = discover_deployment(tmp_path)

    assert len(deployment.casts) == 3
    assert [c.info.file for c in deployment.casts] == [
        "WISE_CAST_001_190817_220856_URC.csv",
        "WISE_CAST_002_190817_221224_URC.csv",
        "WISE_CAST_003_190817_221636_URC.csv",
    ]


def test_discover_deployment_applies_select_flags(tmp_path):
    write_deployment(tmp_path)
    deployment = discover_deployment(tmp_path)

    by_file = {c.info.file: c for c in deployment.casts}
    assert by_file["WISE_CAST_001_190817_220856_URC.csv"].selection.flag == 1
    assert by_file["WISE_CAST_001_190817_220856_URC.csv"].kept is True
    assert by_file["WISE_CAST_002_190817_221224_URC.csv"].selection.flag == 0
    assert by_file["WISE_CAST_002_190817_221224_URC.csv"].kept is False


def test_discover_deployment_defaults_missing_select_row_to_kept(tmp_path):
    write_deployment(tmp_path)
    deployment = discover_deployment(tmp_path)

    cast_003 = next(c for c in deployment.casts if c.info.file == "WISE_CAST_003_190817_221636_URC.csv")
    assert cast_003.selection.flag == 1
    assert cast_003.kept is True


def test_discover_deployment_without_select_file_keeps_everything(tmp_path):
    write_deployment(tmp_path, with_select=False)
    deployment = discover_deployment(tmp_path)

    assert len(deployment.kept_casts()) == 3


def test_kept_casts_filters_rejected(tmp_path):
    write_deployment(tmp_path)
    deployment = discover_deployment(tmp_path)

    kept_files = {c.info.file for c in deployment.kept_casts()}
    assert kept_files == {
        "WISE_CAST_001_190817_220856_URC.csv",
        "WISE_CAST_003_190817_221636_URC.csv",
    }


def test_read_deployment_casts_only_kept(tmp_path):
    write_deployment(tmp_path)
    deployment = discover_deployment(tmp_path)

    result = read_deployment_casts(deployment)

    assert set(result.datasets) == {
        "WISE_CAST_001_190817_220856_URC.csv",
        "WISE_CAST_003_190817_221636_URC.csv",
    }
    assert result.failures == []
    ds = result.datasets["WISE_CAST_001_190817_220856_URC.csv"]
    assert ds.attrs["longitude"] == -68.11626
    assert ds.attrs["latitude"] == 49.24872
    assert ds.attrs["chl_flag"] == 0.0
    assert ds.attrs["qc_flag"] == 1
    assert ds.attrs["rrs_method"] == "Rrs.0p"
    assert ds.sizes["time"] == 3


def test_read_deployment_casts_all(tmp_path):
    write_deployment(tmp_path)
    deployment = discover_deployment(tmp_path)

    result = read_deployment_casts(deployment, only_kept=False)

    assert len(result.datasets) == 3
    assert result.failures == []


def test_read_deployment_casts_sets_shallow_attr(tmp_path):
    write_deployment(tmp_path)
    deployment = discover_deployment(tmp_path)

    result = read_deployment_casts(deployment, only_kept=False)

    # write_deployment()'s SELECT_COPS_DAT fixture has no shallow-flagged casts
    for ds in result.datasets.values():
        assert ds.attrs["shallow"] is False


def test_read_deployment_casts_sets_time_window_attr(tmp_path):
    write_deployment(tmp_path)
    deployment = discover_deployment(tmp_path)

    result = read_deployment_casts(deployment, only_kept=False)

    # cast 003 has "0,90" in info.cops.dat; casts 001/002 have "x" (no override).
    assert result.datasets["WISE_CAST_003_190817_221636_URC.csv"].attrs["time_window"] == (0.0, 90.0)
    assert result.datasets["WISE_CAST_001_190817_220856_URC.csv"].attrs["time_window"] is None
    assert result.datasets["WISE_CAST_002_190817_221224_URC.csv"].attrs["time_window"] is None


def test_read_deployment_casts_continues_after_one_bad_cast(tmp_path, recwarn):
    write_deployment_with_bad_cast(tmp_path)
    deployment = discover_deployment(tmp_path)

    result = read_deployment_casts(deployment, only_kept=False)

    assert set(result.datasets) == {
        "WISE_CAST_001_190817_220856_URC.csv",
        "WISE_CAST_003_190817_221636_URC.csv",
    }
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.file == "WISE_CAST_002_notadate_notatime_URC.csv"
    assert failure.error
    assert any("failed to read cast" in str(w.message) for w in recwarn.list)
