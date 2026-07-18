from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

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
    assert info.cast_number == "001"
    assert info.gps_file == tmp_path / "COPS_IML4_150626_1546_gps.tsv"


def test_parse_cast_filename_legacy_one_token(tmp_path):
    cast_path = tmp_path / "ARCN11_110801_1957_data_002.tsv"
    cast_path.write_text("dummy\n")

    info = parse_cast_filename(cast_path, number_of_fields_before_date=1)

    assert info.date == datetime(2011, 8, 1, 19, 57)
    assert info.is_urc is False
    assert info.cast_number == "002"


def test_parse_cast_filename_autodetects_without_hint(tmp_path):
    cast_path = write_urc_cast(tmp_path)
    info = parse_cast_filename(cast_path)

    assert info.date == datetime(2019, 8, 17, 22, 8)
    assert info.cast_number == "001"
    assert info.gps_file == tmp_path / "GPS_190817.tsv"


def test_parse_cast_filename_wrong_hint_warns_and_still_parses(tmp_path):
    # Reproduces a real bug found on sabre: two sibling WISEMan station folders
    # have identically-shaped filenames, but one folder's init.cops.dat says
    # number.of.fields.before.date=3 (correct) and another says 4 (wrong).
    cast_path = write_urc_cast(tmp_path)

    with pytest.warns(UserWarning, match="number.of.fields.before.date"):
        info = parse_cast_filename(cast_path, number_of_fields_before_date=4)

    assert info.date == datetime(2019, 8, 17, 22, 8)
    assert info.cast_number == "001"


def test_parse_cast_filename_many_tokens_with_and_without_hint(tmp_path):
    cast_path = tmp_path / "AN2303_AOP_BRG_AC23_CAST_002_230913_193250_URC.tsv"
    cast_path.write_text("dummy\n")

    with_hint = parse_cast_filename(cast_path, number_of_fields_before_date=6)
    without_hint = parse_cast_filename(cast_path)

    for info in (with_hint, without_hint):
        assert info.date == datetime(2023, 9, 13, 19, 32)
        assert info.cast_number == "002"


def test_parse_cast_filename_ambiguous_resolved_by_valid_hint(tmp_path):
    # Two plausible date/time pairs: tokens (1,2) and (3,4).
    cast_path = tmp_path / "SITE_190817_2208_190818_2209_URC.csv"
    cast_path.write_text("dummy\n")

    first = parse_cast_filename(cast_path, number_of_fields_before_date=1)
    second = parse_cast_filename(cast_path, number_of_fields_before_date=3)

    assert first.date == datetime(2019, 8, 17, 22, 8)
    assert second.date == datetime(2019, 8, 18, 22, 9)


def test_parse_cast_filename_ambiguous_without_hint_raises(tmp_path):
    cast_path = tmp_path / "SITE_190817_2208_190818_2209_URC.csv"
    cast_path.write_text("dummy\n")

    with pytest.raises(ValueError, match="ambiguous"):
        parse_cast_filename(cast_path)


def test_parse_cast_filename_ambiguous_with_invalid_hint_raises(tmp_path):
    cast_path = tmp_path / "SITE_190817_2208_190818_2209_URC.csv"
    cast_path.write_text("dummy\n")

    with pytest.raises(ValueError, match="ambiguous"):
        parse_cast_filename(cast_path, number_of_fields_before_date=99)


def test_parse_cast_filename_unparseable_raises(tmp_path):
    cast_path = tmp_path / "WISE_summary_report.csv"
    cast_path.write_text("dummy\n")

    with pytest.raises(ValueError, match="could not find"):
        parse_cast_filename(cast_path)


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


def test_read_cast_default_hint_none_autodetects(tmp_path):
    cast_path = write_urc_cast(tmp_path)
    ds = read_cast(cast_path)  # no number_of_fields_before_date at all

    assert ds.attrs["cast_number"] == "001"
    assert ds.attrs["cast_date"] == datetime(2019, 8, 17, 22, 8).isoformat()


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
