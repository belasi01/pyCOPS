from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pycops.processing.position import find_gps_file, position_from_gps, read_gps_file

GPS_TSV = """\
"[DateTime]"\t"DateTimeUTC"\t"Millisecond"\t"[GpsTime]"\t"Latitude"\t"Longitude"\t"SatelliteCount"
"6/5/2018 6:36:37 PM"\t"06/05/2018 06:36:37.116 PM"\t0\t"6/5/2018 6:35:42 PM"\t63.1770\t-81.8483\t11
"6/5/2018 6:41:00 PM"\t"06/05/2018 06:41:00 PM"\t0\t"6/5/2018 6:40:59 PM"\t63.1772\t-81.8481\t9
"6/5/2018 6:45:22 PM"\t"06/05/2018 06:45:22.500 PM"\t500\t"6/5/2018 6:45:21 PM"\t63.1774\t-81.8479\t10
"""


def test_read_gps_file_parses_real_column_layout(tmp_path):
    path = tmp_path / "GPS_180605.tsv"
    path.write_text(GPS_TSV)

    table = read_gps_file(path)

    assert list(table.columns) == ["time", "longitude", "latitude"]
    assert len(table) == 3
    assert table["time"].dtype.kind == "M"  # datetime64
    np.testing.assert_allclose(table["longitude"], [-81.8483, -81.8481, -81.8479])
    np.testing.assert_allclose(table["latitude"], [63.1770, 63.1772, 63.1774])


def test_read_gps_file_handles_inconsistent_millisecond_precision(tmp_path):
    # real files mix "...:37.116 PM" and "...:00 PM" (no fractional seconds) in the same column
    path = tmp_path / "GPS_180605.tsv"
    path.write_text(GPS_TSV)

    table = read_gps_file(path)

    assert table["time"].is_monotonic_increasing


def test_read_gps_file_drops_repeated_header_mid_file(tmp_path):
    # Simulates a GPS/COPS logger restart partway through the day: uProfile
    # re-emits the header row (twice here, to also cover multiple restarts).
    header = GPS_TSV.splitlines()[0]
    lines = GPS_TSV.splitlines()
    corrupted = "\n".join([lines[0], lines[1], header, lines[2], header, lines[3]]) + "\n"
    path = tmp_path / "GPS_180605.tsv"
    path.write_text(corrupted)

    with pytest.warns(UserWarning, match="dropped 2 repeated header row"):
        table = read_gps_file(path)

    assert len(table) == 3
    np.testing.assert_allclose(table["longitude"], [-81.8483, -81.8481, -81.8479])
    np.testing.assert_allclose(table["latitude"], [63.1770, 63.1772, 63.1774])


def test_position_from_gps_median_within_cast_window(tmp_path):
    path = tmp_path / "GPS_180605.tsv"
    path.write_text(GPS_TSV)
    table = read_gps_file(path)

    cast_times = pd.date_range("2018-06-05T18:40:00", "2018-06-05T18:46:00", periods=5).values
    result = position_from_gps(table, cast_times)

    assert result is not None
    longitude, latitude = result
    # median of the 2nd and 3rd GPS fixes, the only ones inside the window
    assert longitude == pytest.approx((-81.8481 + -81.8479) / 2)
    assert latitude == pytest.approx((63.1772 + 63.1774) / 2)


def test_position_from_gps_returns_none_when_no_overlap(tmp_path):
    path = tmp_path / "GPS_180605.tsv"
    path.write_text(GPS_TSV)
    table = read_gps_file(path)

    cast_times = pd.date_range("2018-06-06T00:00:00", "2018-06-06T00:10:00", periods=5).values
    assert position_from_gps(table, cast_times) is None


def test_find_gps_file_finds_gps_prefixed_file(tmp_path):
    (tmp_path / "info.cops.dat").write_text("")
    (tmp_path / "GPS_180605.tsv").write_text(GPS_TSV)

    found = find_gps_file(tmp_path)
    assert found is not None
    assert found.name == "GPS_180605.tsv"


def test_find_gps_file_none_when_absent(tmp_path):
    (tmp_path / "info.cops.dat").write_text("")
    assert find_gps_file(tmp_path) is None
