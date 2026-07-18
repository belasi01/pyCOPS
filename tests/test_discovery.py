from __future__ import annotations

from conftest import write_deployment
from pycops.io.discovery import discover_deployment, read_deployment_casts


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

    datasets = read_deployment_casts(deployment)

    assert set(datasets) == {
        "WISE_CAST_001_190817_220856_URC.csv",
        "WISE_CAST_003_190817_221636_URC.csv",
    }
    ds = datasets["WISE_CAST_001_190817_220856_URC.csv"]
    assert ds.attrs["longitude"] == -68.11626
    assert ds.attrs["latitude"] == 49.24872
    assert ds.attrs["chl_flag"] == 0.0
    assert ds.attrs["qc_flag"] == 1
    assert ds.attrs["rrs_method"] == "Rrs.0p"
    assert ds.sizes["time"] == 3


def test_read_deployment_casts_all(tmp_path):
    write_deployment(tmp_path)
    deployment = discover_deployment(tmp_path)

    datasets = read_deployment_casts(deployment, only_kept=False)

    assert len(datasets) == 3
