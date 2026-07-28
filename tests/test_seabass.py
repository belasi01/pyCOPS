from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pycops.io.seabass import SeaBASSHeaderFields, write_seabass_station_file
from pycops.processing.database import MeanSd, StationAggregate

WAVES = np.array([443.0, 555.0])


def _make_aggregate(rrs_mean=(1.0, np.nan), rrs_sd=(0.1, 0.2)):
    n = len(WAVES)
    zeros = MeanSd(mean=np.zeros(n), sd=np.zeros(n))
    return StationAggregate(
        station_id="MAN-F05",
        directory=Path("/data/MAN-F05"),
        n_casts=3,
        date_mean=pd.Timestamp("2019-08-18T18:18:35"),
        sun_zenith_mean=45.5,
        longitude_mean=-68.108833,
        latitude_mean=49.13445,
        forel_ule_mean=6.0,
        shadow_correction_method="abs.Kd.method",
        bottom_depth_mean=5.2,
        ed0_diffuse_fraction=np.array([0.4, 0.45]),
        rrs=MeanSd(mean=np.asarray(rrs_mean, dtype=float), sd=np.asarray(rrs_sd, dtype=float)),
        nlw=zeros,
        rb=MeanSd(mean=np.full(n, np.nan), sd=np.full(n, np.nan)),
        kd_1pct=zeros,
        kd_10pct=zeros,
        kd_pd=zeros,
        ed0_0p=MeanSd(mean=np.array([100.0, 200.0]), sd=np.array([1.0, 2.0])),
    )


def _read_sb(path):
    lines = path.read_text().splitlines()
    header_end = lines.index("/end_header")
    header = lines[:header_end]
    data_row = lines[header_end + 1]
    return header, data_row


def test_write_seabass_station_file_header_has_documented_keys(tmp_path):
    header = SeaBASSHeaderFields(investigators="Belanger,S", affiliations="UQAR", experiment="WISEMan")
    path = tmp_path / "MAN-F05.sb"

    write_seabass_station_file(_make_aggregate(), header, WAVES, path)

    lines, _ = _read_sb(path)
    assert lines[0] == "/begin_header"
    assert "/investigators=Belanger,S" in lines
    assert "/affiliations=UQAR" in lines
    assert "/experiment=WISEMan" in lines
    assert "/station=MAN-F05" in lines
    assert f"/data_file_name={path.name}" in lines
    assert "/start_date=20190818" in lines
    assert "/start_time=18:18:35" in lines
    assert "/missing=-9999" in lines
    assert "/delimiter=comma" in lines


def test_write_seabass_station_file_fields_and_units_lengths_match(tmp_path):
    path = tmp_path / "MAN-F05.sb"
    write_seabass_station_file(_make_aggregate(), SeaBASSHeaderFields(), WAVES, path)

    lines, _ = _read_sb(path)
    fields_line = next(line for line in lines if line.startswith("/fields="))
    units_line = next(line for line in lines if line.startswith("/units="))
    fields = fields_line[len("/fields=") :].split(",")
    units = units_line[len("/units=") :].split(",")

    assert len(fields) == len(units)
    assert "Rrs443" in fields
    assert "Rrs443_sd" in fields
    assert "Kd1pct443" in fields  # non-standard, but present and clearly named


def test_write_seabass_station_file_nan_becomes_missing_sentinel(tmp_path):
    path = tmp_path / "MAN-F05.sb"
    header = SeaBASSHeaderFields(missing=-9999)
    write_seabass_station_file(_make_aggregate(rrs_mean=(1.0, np.nan)), header, WAVES, path)

    lines, data_row = _read_sb(path)
    fields_line = next(line for line in lines if line.startswith("/fields="))
    fields = fields_line[len("/fields=") :].split(",")
    values = data_row.split(",")

    rrs555_value = values[fields.index("Rrs555")]
    rrs443_value = values[fields.index("Rrs443")]
    assert rrs555_value == "-9999"
    assert rrs443_value == "1"


def test_write_seabass_station_file_data_row_matches_fields_length(tmp_path):
    path = tmp_path / "MAN-F05.sb"
    write_seabass_station_file(_make_aggregate(), SeaBASSHeaderFields(), WAVES, path)

    lines, data_row = _read_sb(path)
    fields_line = next(line for line in lines if line.startswith("/fields="))
    fields = fields_line[len("/fields=") :].split(",")
    values = data_row.split(",")

    assert len(values) == len(fields)


def test_write_seabass_station_file_notes_non_standard_kd_fields(tmp_path):
    path = tmp_path / "MAN-F05.sb"
    write_seabass_station_file(_make_aggregate(), SeaBASSHeaderFields(), WAVES, path)

    lines, _ = _read_sb(path)
    assert any(line.startswith("!") and "Kd1pct" in line for line in lines)
