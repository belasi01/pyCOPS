from __future__ import annotations

from datetime import datetime

import numpy as np

from conftest import write_urc_cast
from pycops.io.raw import _clean_column_names, parse_cast_filename, read_cast


def test_parse_cast_filename_urc(tmp_path):
    cast_path = write_urc_cast(tmp_path)
    info = parse_cast_filename(cast_path, number_of_fields_before_date=3)

    assert info.date == datetime(2019, 8, 17, 22, 8)
    assert info.cast_number == "001"
    assert info.is_urc is True
    assert info.gps_file == tmp_path / "GPS_190817.tsv"


def test_parse_cast_filename_legacy(tmp_path):
    cast_path = tmp_path / "COPS_IML4_150626_1546_C_data_001.csv"
    cast_path.write_text("dummy\n")
    (tmp_path / "COPS_IML4_150626_1546_gps.tsv").write_text("dummy\n")

    info = parse_cast_filename(cast_path, number_of_fields_before_date=2)

    assert info.date == datetime(2015, 6, 26, 15, 46)
    assert info.is_urc is False
    assert info.gps_file == tmp_path / "COPS_IML4_150626_1546_gps.tsv"


def test_parse_cast_filename_no_gps_file(tmp_path):
    cast_path = write_urc_cast(tmp_path)
    (tmp_path / "GPS_190817.tsv").unlink()

    info = parse_cast_filename(cast_path, number_of_fields_before_date=3)

    assert info.gps_file is None


def test_parse_cast_filename_bioshade(tmp_path):
    # BioShade casts are named "<site>_SB_<date>_<time>_URC.<ext>" -- one fewer
    # token before the date than "<site>_CAST_NNN_<date>_<time>_URC.<ext>".
    cast_path = tmp_path / "hudsonbay_SB_180605_192518_URC.csv"
    cast_path.write_text("dummy\n")
    (tmp_path / "GPS_180605.tsv").write_text("dummy\n")

    info = parse_cast_filename(cast_path, number_of_fields_before_date=3)

    assert info.date == datetime(2018, 6, 5, 19, 25)
    assert info.cast_number == "SB"
    assert info.is_urc is True
    assert info.gps_file == tmp_path / "GPS_180605.tsv"


def test_clean_column_names_strips_units():
    assert _clean_column_names(["Ed0443 (uW/(cm2 nm))", "LuZTemp (degC)"]) == ["Ed0443", "LuZTemp"]


def test_clean_column_names_legacy_brackets():
    assert _clean_column_names(["Ed0[340]", "Ed0[380]"]) == ["Ed0340", "Ed0380"]


def test_read_cast_basic_structure(tmp_path):
    cast_path = write_urc_cast(tmp_path)
    ds = read_cast(cast_path, number_of_fields_before_date=3)

    assert ds.sizes["time"] == 3
    assert list(ds["wavelength"].values) == [320.0, 340.0, 443.0]
    assert ds["Ed0"].dims == ("time", "wavelength")
    assert ds["Ed0"].shape == (3, 3)

    np.testing.assert_allclose(ds["Ed0"].isel(time=0).values, [2.25, 6.04, 19.29])


def test_read_cast_ancillary_and_others(tmp_path):
    cast_path = write_urc_cast(tmp_path)
    ds = read_cast(cast_path, number_of_fields_before_date=3)

    assert "Ed0_Roll" in ds
    assert "LuZ_Depth" in ds
    assert "LuZ_Temp" in ds
    np.testing.assert_allclose(ds["LuZ_Depth"].values, [0.098, 0.092, 0.093])

    assert "BioGPSPosition" in ds
    assert "GeneralExcelTime" in ds


def test_read_cast_time_from_datetime_utc(tmp_path):
    cast_path = write_urc_cast(tmp_path)
    ds = read_cast(cast_path, number_of_fields_before_date=3)

    times = ds["time"].values
    assert times.dtype.kind == "M"
    assert times[0] < times[1] < times[2]


def test_read_cast_skips_header_block(tmp_path):
    cast_path = write_urc_cast(tmp_path, with_header_block=True)
    ds = read_cast(cast_path, number_of_fields_before_date=3)

    assert ds.sizes["time"] == 3
    np.testing.assert_allclose(ds["Ed0"].isel(time=0).values, [2.25, 6.04, 19.29])


def test_read_cast_attrs(tmp_path):
    cast_path = write_urc_cast(tmp_path)
    ds = read_cast(cast_path, number_of_fields_before_date=3)

    assert ds.attrs["cast_number"] == "001"
    assert ds.attrs["is_urc_format"] is True
    assert ds.attrs["instruments"] == ["Ed0", "EdZ", "LuZ", "EuZ"]
    assert ds.attrs["gps_file"].endswith("GPS_190817.tsv")


def test_read_cast_bioshade_forces_ed0_only(tmp_path):
    header = "DateTime,Ed0320 (uW/(cm2 nm)),BioShade_Position (position)"
    rows = [
        "06/05/2018 19:25:19,37.15,25550",
        "06/05/2018 19:25:20,38.35,25600",
    ]
    path = tmp_path / "hudsonbay_SB_180605_192518_URC.csv"
    path.write_text("\n".join([header, *rows]) + "\n")

    # requesting the usual 4 instruments should still only pick up Ed0
    ds = read_cast(path, number_of_fields_before_date=3)

    assert ds.attrs["instruments"] == ["Ed0"]
    assert ds.attrs["cast_number"] == "SB"
    assert "BioShade_Position" in ds


def test_read_cast_omits_instrument_absent_from_file(tmp_path):
    # Many real deployments only carry Ed0/EdZ/LuZ, with no EuZ sensor at all.
    header = "DateTime,Ed0320 (uW/(cm2 nm)),LuZ320 (uW/(sr cm2 nm)),LuZTemp (degC)"
    rows = [
        "08/17/2019 22:08:59,2.25,4.22e-05,9.39",
        "08/17/2019 22:09:00,2.26,4.23e-05,9.40",
    ]
    path = tmp_path / "WISE_CAST_001_190817_220856_URC.csv"
    path.write_text("\n".join([header, *rows]) + "\n")

    ds = read_cast(path, number_of_fields_before_date=3)

    assert "EuZ" not in ds.data_vars
    assert "wavelength_EuZ" not in ds.dims
    assert ds.attrs["instruments"] == ["Ed0", "LuZ"]
