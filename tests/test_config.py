from __future__ import annotations

import warnings

import numpy as np
import pytest

from conftest import INFO_COPS_DAT, INIT_COPS_DAT
from pycops.io.config import (
    absorption_for_cast,
    default_init_cops_params,
    format_init_cops_dat,
    migrate_init_cops_dat,
    read_absorption_cops,
    read_info_cops,
    read_init_cops,
    update_cast_info,
    update_time_window,
    write_init_cops,
)


def test_read_init_cops_scalars(tmp_path):
    path = tmp_path / "init.cops.dat"
    path.write_text(INIT_COPS_DAT)

    params = read_init_cops(path)

    assert params["verbose"] is True
    assert params["indice.water"] == 1.34
    assert params["instruments.optics"] == ["Ed0", "EdZ", "LuZ", "EuZ"]
    assert params["format.date"] == "%m/%d/%Y %H:%M:%S"
    assert params["number.of.fields.before.date"] == 3.0


def test_read_init_cops_per_instrument_vectors(tmp_path):
    path = tmp_path / "init.cops.dat"
    path.write_text(INIT_COPS_DAT)

    params = read_init_cops(path)

    assert params["tiltmax.optics"] == {"Ed0": 10.0, "EdZ": 5.0, "LuZ": 5.0, "EuZ": 5.0}
    assert params["delta.capteur.optics"] == {"Ed0": 0.0, "EdZ": -0.05, "LuZ": 0.238, "EuZ": 0.238}


def test_read_init_cops_na_sentinel_in_numeric_vector(tmp_path):
    # Real deployments use R's "NA" sentinel for a threshold that doesn't apply
    # to the surface (Ed0) instrument, e.g. linear.fit.Rsquared.threshold.optics.
    content = INIT_COPS_DAT + "linear.fit.Rsquared.threshold.optics;numeric; NA, 0.5, 0.6,0.6\n"
    path = tmp_path / "init.cops.dat"
    path.write_text(content)

    params = read_init_cops(path)

    thresholds = params["linear.fit.Rsquared.threshold.optics"]
    assert thresholds["Ed0"] != thresholds["Ed0"]  # NaN
    assert thresholds["EdZ"] == 0.5
    assert thresholds["LuZ"] == 0.6


def test_read_init_cops_defaults_missing_linear_fit_params(tmp_path):
    # Real init.cops.dat files predating these R package parameters (e.g. the
    # WISEMan 2020 deployments) don't have them at all.
    path = tmp_path / "init.cops.dat"
    path.write_text(INIT_COPS_DAT)

    with pytest.warns(UserWarning, match="linear.fit.Rsquared.threshold.optics"):
        params = read_init_cops(path)

    r2 = params["linear.fit.Rsquared.threshold.optics"]
    assert r2["EdZ"] == 0.5
    assert r2["LuZ"] == 0.6
    assert r2["EuZ"] == 0.6
    assert r2["Ed0"] != r2["Ed0"]  # NaN

    max_delta = params["linear.fit.max.delta.depth.optics"]
    assert max_delta["Ed0"] != max_delta["Ed0"]  # NaN
    assert max_delta["EdZ"] == 3.0
    assert max_delta["LuZ"] == 2.5
    assert max_delta["EuZ"] == 2.5


def test_read_init_cops_defaults_missing_windspeed(tmp_path):
    path = tmp_path / "init.cops.dat"
    path.write_text(INIT_COPS_DAT)

    with pytest.warns(UserWarning, match="windspeed_ms"):
        params = read_init_cops(path)

    assert params["windspeed_ms"] == 4.0


def test_read_init_cops_does_not_override_present_windspeed(tmp_path):
    content = INIT_COPS_DAT + "windspeed_ms;numeric;9.8\n"
    path = tmp_path / "init.cops.dat"
    path.write_text(content)

    params = read_init_cops(path)

    assert params["windspeed_ms"] == 9.8


def test_read_init_cops_parses_ed0_correction_method(tmp_path):
    content = INIT_COPS_DAT + "ed0.correction.method;character;smoothed\n"
    path = tmp_path / "init.cops.dat"
    path.write_text(content)

    params = read_init_cops(path)

    assert params["ed0.correction.method"] == "smoothed"


def test_read_init_cops_defaults_missing_ed0_correction_method(tmp_path):
    # Real init.cops.dat files predate this pycops-only field entirely.
    path = tmp_path / "init.cops.dat"
    path.write_text(INIT_COPS_DAT)

    with pytest.warns(UserWarning, match="ed0.correction.method"):
        params = read_init_cops(path)

    assert params["ed0.correction.method"] == "raw"


def test_default_init_cops_params_round_trips_ed0_correction_method(tmp_path):
    params = default_init_cops_params(["Ed0", "EdZ", "LuZ", "EuZ"])
    assert params["ed0.correction.method"] == "raw"

    path = tmp_path / "init.cops.dat"
    write_init_cops(path, params)
    reread = read_init_cops(path)

    assert reread["ed0.correction.method"] == "raw"


def test_format_init_cops_dat_writes_ed0_correction_method_line(tmp_path):
    params = default_init_cops_params(["Ed0", "EdZ", "LuZ", "EuZ"])
    params["ed0.correction.method"] = "smoothed"

    text = format_init_cops_dat(params)

    assert "ed0.correction.method;character;smoothed" in text


def test_format_init_cops_dat_round_trips_legacy_time_interval_field(tmp_path):
    """A still-unmigrated legacy init.cops.dat (time.interval.for.smoothing.optics, no
    depth.interval.for.smoothing.optics) read directly with read_init_cops() -- bypassing
    migrate_init_cops_dat() -- must still round-trip both fields through
    format_init_cops_dat/write_init_cops rather than silently dropping the legacy one; depth.
    interval.for.smoothing.optics is now also present, in-memory, with its own standard default
    (matching every other backfilled parameter -- see _PER_INSTRUMENT_DEFAULTS)."""
    content = INIT_COPS_DAT.replace(
        "depth.interval.for.smoothing.optics;numeric; 10, 4,4,4\n",
        "time.interval.for.smoothing.optics;numeric; 40, 40, 20,20\n",
    )
    path = tmp_path / "init.cops.dat"
    path.write_text(content)
    params = read_init_cops(path)
    assert params["time.interval.for.smoothing.optics"] == {"Ed0": 40.0, "EdZ": 40.0, "LuZ": 20.0, "EuZ": 20.0}
    assert params["depth.interval.for.smoothing.optics"] == {"Ed0": 10.0, "EdZ": 5.0, "LuZ": 5.0, "EuZ": 5.0}

    text = format_init_cops_dat(params)

    assert "time.interval.for.smoothing.optics;numeric;40,40,20,20" in text
    assert "depth.interval.for.smoothing.optics;numeric;10,5,5,5" in text

    reread_path = tmp_path / "init_rewritten.cops.dat"
    reread_path.write_text(text, newline="")
    reread = read_init_cops(reread_path)
    assert reread["time.interval.for.smoothing.optics"] == {"Ed0": 40.0, "EdZ": 40.0, "LuZ": 20.0, "EuZ": 20.0}


# ---- migrate_init_cops_dat: persisting legacy-file default injection to disk ----


def test_migrate_init_cops_dat_appends_missing_fields_and_reports_them(tmp_path):
    # INIT_COPS_DAT (see conftest.py) has bandwidth but is missing linear.fit.Rsquared.threshold
    # .optics/linear.fit.max.delta.depth.optics/windspeed_ms/ed0.correction.method -- exactly the
    # real-world GreenEdge 2016 shape this feature was built for.
    path = tmp_path / "init.cops.dat"
    path.write_text(INIT_COPS_DAT)

    with pytest.warns(UserWarning):  # read_init_cops's own warnings still fire underneath
        changes = migrate_init_cops_dat(path)

    assert len(changes) == 4
    assert any("linear.fit.Rsquared.threshold.optics" in c for c in changes)
    assert any("linear.fit.max.delta.depth.optics" in c for c in changes)
    assert any("windspeed_ms" in c for c in changes)
    assert any("ed0.correction.method" in c for c in changes)

    text = path.read_text()
    assert "linear.fit.Rsquared.threshold.optics;numeric;" in text
    assert "windspeed_ms;numeric;4" in text
    assert "ed0.correction.method;character;raw" in text
    # the original content must round-trip untouched, not just the new lines
    assert "instruments.optics;character;Ed0,EdZ,LuZ,EuZ" in text

    # any write also backs up the pre-migration file, matching Simon's general request that no
    # legacy init.cops.dat is ever edited without a preserved original copy.
    backup = tmp_path / "legacy.init.cops.dat"
    assert backup.exists()
    assert backup.read_text() == INIT_COPS_DAT


def test_migrate_init_cops_dat_backup_created_only_once(tmp_path):
    """A second, later migration (e.g. once a genuinely new field is added to
    _PER_INSTRUMENT_DEFAULTS/_SCALAR_DEFAULTS down the road) must not overwrite an existing
    legacy.init.cops.dat with an already-partially-migrated version -- the backup should always
    hold the oldest, truly original file."""
    path = tmp_path / "init.cops.dat"
    path.write_text(INIT_COPS_DAT)
    migrate_init_cops_dat(path)
    backup = tmp_path / "legacy.init.cops.dat"
    assert backup.read_text() == INIT_COPS_DAT

    # simulate the file being hand-edited after its first migration, then losing a field again
    path.write_text(path.read_text().replace("windspeed_ms;numeric;4\n", ""))
    migrate_init_cops_dat(path)

    assert backup.read_text() == INIT_COPS_DAT  # still the very first backup, untouched


def test_migrate_init_cops_dat_replaces_legacy_time_interval_with_default_depth_values(tmp_path):
    """Simon's explicit follow-up (2026-08-19): don't derive depth.interval.for.smoothing.optics
    from the old per-cast time.interval.for.smoothing.optics values at all -- comment out the
    legacy field and use the same standard default values as every other backfilled parameter."""
    content = INIT_COPS_DAT.replace(
        "depth.interval.for.smoothing.optics;numeric; 10, 4,4,4\n",
        "time.interval.for.smoothing.optics;numeric; 40, 40, 20,20\n",
    )
    path = tmp_path / "init.cops.dat"
    path.write_text(content)

    changes = migrate_init_cops_dat(path)

    assert any(
        "time.interval.for.smoothing.optics" in c and "depth.interval.for.smoothing.optics" in c
        for c in changes
    )
    text = path.read_text()
    assert "# time.interval.for.smoothing.optics;numeric; 40, 40, 20,20" in text
    assert "depth.interval.for.smoothing.optics;numeric;10,5,5,5" in text  # standard defaults, not 40/40/20-derived

    reread = read_init_cops(path)
    assert reread["depth.interval.for.smoothing.optics"] == {"Ed0": 10.0, "EdZ": 5.0, "LuZ": 5.0, "EuZ": 5.0}
    assert "time.interval.for.smoothing.optics" not in reread  # the commented-out line isn't parsed

    backup = tmp_path / "legacy.init.cops.dat"
    assert backup.read_text() == content  # original, uncommented time.interval line preserved


def test_migrate_init_cops_dat_is_idempotent(tmp_path):
    path = tmp_path / "init.cops.dat"
    path.write_text(INIT_COPS_DAT)
    migrate_init_cops_dat(path)

    second_pass = migrate_init_cops_dat(path)

    assert second_pass == []


def test_migrate_init_cops_dat_leaves_read_init_cops_warning_free_afterward(tmp_path):
    path = tmp_path / "init.cops.dat"
    path.write_text(INIT_COPS_DAT)
    migrate_init_cops_dat(path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warnings.warn() now fails the test
        params = read_init_cops(path)

    assert params["windspeed_ms"] == 4.0
    assert params["ed0.correction.method"] == "raw"


def test_migrate_init_cops_dat_no_op_on_a_complete_file(tmp_path):
    params = default_init_cops_params(["Ed0", "EdZ", "LuZ", "EuZ"])
    path = tmp_path / "init.cops.dat"
    write_init_cops(path, params)

    changes = migrate_init_cops_dat(path)

    assert changes == []


def test_migrate_init_cops_dat_preserves_crlf_terminator(tmp_path):
    path = tmp_path / "init.cops.dat"
    path.write_bytes(INIT_COPS_DAT.replace("\n", "\r\n").encode())

    migrate_init_cops_dat(path)

    data = path.read_bytes()
    assert b"\r\n" in data
    # every line in the file (original + newly appended) uses the same CRLF convention
    assert data.count(b"\n") == data.count(b"\r\n")


def test_read_info_cops(tmp_path):
    path = tmp_path / "info.cops.dat"
    path.write_text(INFO_COPS_DAT)

    entries = read_info_cops(path)

    assert len(entries) == 3

    first = entries[0]
    assert first.file == "WISE_CAST_001_190817_220856_URC.csv"
    assert first.longitude == -68.11626
    assert first.chl_flag == 0.0
    assert first.time_window is None
    assert first.dark_files == []

    second = entries[1]
    assert second.chl_flag is None  # "NA" means no shadow correction

    third = entries[2]
    assert third.time_window == (0.0, 90.0)
    assert third.sub_surface_removed_layer == [0.1, 0.05, 0.1, 0.0]
    assert third.linear_r2_threshold == [0.5, 0.6, 0.5, 0.6]
    assert third.linear_max_delta_depth == [3.0, 3.0, 2.5, 2.5]
    assert third.dark_files == ["dark_001.csv"]


def test_read_info_cops_na_within_override_field(tmp_path):
    # Real AlgaeWISE PME_CAST_019 row has "NA,0.5,0.5,0.6" for linear.fit.Rsquared.threshold
    # (Ed0's threshold doesn't apply at the surface) -- same per-value NA sentinel init.cops.dat
    # uses for its own per-instrument vectors.
    path = tmp_path / "info.cops.dat"
    path.write_text("PME_CAST_019_220705_152602_URC.csv;-64.3476;49.7745;999;x;x;10,5,7,5;x;NA,0.5,0.5,0.6;x\n")

    entries = read_info_cops(path)

    r2 = entries[0].linear_r2_threshold
    assert r2[1:] == [0.5, 0.5, 0.6]
    assert np.isnan(r2[0])


def test_read_info_cops_blank_chl_field_is_none_not_a_crash(tmp_path):
    # A short row (fewer than 4 fields) pads chl out to "" via read_info_cops's own padding --
    # that must parse the same as an explicit "NA", not raise trying float("").
    path = tmp_path / "info.cops.dat"
    path.write_text("WISE_CAST_001_190817_220856_URC.csv;-68.11626;49.24872\n")

    entries = read_info_cops(path)

    assert entries[0].chl_flag is None


def test_read_info_cops_na_longitude_latitude(tmp_path):
    # Real BaySYS station015 data has "NA" lon/lat -- position instead comes
    # from a GPS_*.tsv file (not yet ported); this must not crash discovery.
    path = tmp_path / "info.cops.dat"
    path.write_text("hudsonbay_CAST_001_180605_194923_URC.csv;NA;NA;999;x;x;x;x\n")

    entries = read_info_cops(path)

    assert entries[0].longitude is None
    assert entries[0].latitude is None


def test_update_time_window_creates_missing_file(tmp_path):
    path = tmp_path / "info.cops.dat"
    update_time_window(path, "WISE_CAST_001_190817_220856_URC.csv", (10.0, 90.5))

    with path.open(newline="") as f:  # Path.read_text(newline=...) needs Python 3.13+
        text = f.read()
    assert "\r\n" in text  # matches a real cops.go()-generated file
    assert "this file is a table with a maximum of 12 fields" in text  # header block present

    entries = read_info_cops(path)
    assert len(entries) == 1
    assert entries[0].file == "WISE_CAST_001_190817_220856_URC.csv"
    assert entries[0].time_window == (10.0, 90.5)
    assert entries[0].longitude is None


def test_update_time_window_updates_existing_row_preserving_other_lines(tmp_path):
    path = tmp_path / "info.cops.dat"
    path.write_text(INFO_COPS_DAT)
    before = path.read_text()

    update_time_window(path, "WISE_CAST_001_190817_220856_URC.csv", (5.0, 45.0))

    after_lines = path.read_text().splitlines()
    before_lines = before.splitlines()
    changed = [i for i in range(len(before_lines)) if before_lines[i] != after_lines[i]]
    assert len(changed) == 1  # only the target row's line changed

    entries = read_info_cops(path)
    updated = next(e for e in entries if e.file == "WISE_CAST_001_190817_220856_URC.csv")
    assert updated.time_window == (5.0, 45.0)
    # untouched fields on the same row stay as they were
    third = next(e for e in entries if e.file == "WISE_CAST_003_190817_221636_URC.csv")
    assert third.time_window == (0.0, 90.0)
    assert third.dark_files == ["dark_001.csv"]


def test_update_time_window_overwrites_existing_override(tmp_path):
    path = tmp_path / "info.cops.dat"
    path.write_text(INFO_COPS_DAT)

    update_time_window(path, "WISE_CAST_003_190817_221636_URC.csv", (1.0, 2.0))

    entries = read_info_cops(path)
    third = next(e for e in entries if e.file == "WISE_CAST_003_190817_221636_URC.csv")
    assert third.time_window == (1.0, 2.0)
    assert third.sub_surface_removed_layer == [0.1, 0.05, 0.1, 0.0]  # untouched


def test_update_time_window_appends_row_when_cast_not_listed(tmp_path):
    path = tmp_path / "info.cops.dat"
    path.write_text(INFO_COPS_DAT)

    update_time_window(path, "WISE_CAST_999_190817_235959_URC.csv", (0.0, 10.0))

    entries = read_info_cops(path)
    assert len(entries) == 4
    new_entry = next(e for e in entries if e.file == "WISE_CAST_999_190817_235959_URC.csv")
    assert new_entry.time_window == (0.0, 10.0)
    assert new_entry.longitude is None


def test_update_time_window_pads_short_row(tmp_path):
    path = tmp_path / "info.cops.dat"
    path.write_text("WISE_CAST_001_190817_220856_URC.csv;-68.1;49.2;0\n")

    update_time_window(path, "WISE_CAST_001_190817_220856_URC.csv", (3.0, 4.0))

    entries = read_info_cops(path)
    assert entries[0].time_window == (3.0, 4.0)
    assert entries[0].longitude == -68.1


def test_update_time_window_preserves_crlf(tmp_path):
    path = tmp_path / "info.cops.dat"
    crlf_content = INFO_COPS_DAT.replace("\n", "\r\n")
    with path.open("w", newline="") as f:
        f.write(crlf_content)

    update_time_window(path, "WISE_CAST_001_190817_220856_URC.csv", (5.0, 45.0))

    raw = path.read_bytes()
    assert b"\r\n" in raw
    assert b"\n\n" not in raw.replace(b"\r\n", b"")  # no lone LF introduced
    # every line (including the untouched ones) still ends in CRLF
    for line in raw.split(b"\r\n")[:-1]:
        assert b"\n" not in line


def test_update_cast_info_writes_na_sentinel_not_python_nan_repr(tmp_path):
    path = tmp_path / "info.cops.dat"
    update_cast_info(
        path,
        "PME_CAST_019_220705_152602_URC.csv",
        linear_r2_threshold=[float("nan"), 0.5, 0.5, 0.6],
    )

    text = path.read_text()
    assert "NA,0.5,0.5,0.6" in text
    assert "nan" not in text  # no stray lowercase Python "nan" repr


def test_update_cast_info_preserves_coordinate_precision(tmp_path):
    # A plain "%g" format defaults to 6 significant figures and would silently truncate
    # -68.11626 (7 sig figs) to -68.1163 -- real GPS decimal-degree precision needs more.
    path = tmp_path / "info.cops.dat"
    update_cast_info(path, "WISE_CAST_001_190817_220856_URC.csv", longitude=-68.11626, latitude=49.24872)

    entries = read_info_cops(path)
    assert entries[0].longitude == -68.11626
    assert entries[0].latitude == 49.24872


def test_update_cast_info_sets_multiple_fields_at_once(tmp_path):
    path = tmp_path / "info.cops.dat"
    path.write_text(INFO_COPS_DAT)

    update_cast_info(
        path,
        "WISE_CAST_001_190817_220856_URC.csv",
        longitude=1.5,
        latitude=-2.5,
        chl_flag=999.0,
        tiltmax=[5.0, 5.0, 5.0, 5.0],
    )

    entries = read_info_cops(path)
    updated = next(e for e in entries if e.file == "WISE_CAST_001_190817_220856_URC.csv")
    assert updated.longitude == 1.5
    assert updated.latitude == -2.5
    assert updated.chl_flag == 999.0
    assert updated.tiltmax == [5.0, 5.0, 5.0, 5.0]
    assert updated.time_window is None  # untouched (field 5 sits before field 7, still padded "x")


def test_update_cast_info_only_touches_passed_fields(tmp_path):
    path = tmp_path / "info.cops.dat"
    path.write_text(INFO_COPS_DAT)

    update_cast_info(path, "WISE_CAST_003_190817_221636_URC.csv", chl_flag=5.0)

    entries = read_info_cops(path)
    third = next(e for e in entries if e.file == "WISE_CAST_003_190817_221636_URC.csv")
    assert third.chl_flag == 5.0
    # every other field on the same row stays exactly as it was
    assert third.time_window == (0.0, 90.0)
    assert third.sub_surface_removed_layer == [0.1, 0.05, 0.1, 0.0]
    assert third.linear_r2_threshold == [0.5, 0.6, 0.5, 0.6]
    assert third.dark_files == ["dark_001.csv"]


def test_update_cast_info_explicit_none_clears_field(tmp_path):
    path = tmp_path / "info.cops.dat"
    path.write_text(INFO_COPS_DAT)

    update_cast_info(path, "WISE_CAST_003_190817_221636_URC.csv", chl_flag=None, sub_surface_removed_layer=None)

    entries = read_info_cops(path)
    third = next(e for e in entries if e.file == "WISE_CAST_003_190817_221636_URC.csv")
    assert third.chl_flag is None
    assert third.sub_surface_removed_layer is None
    assert third.time_window == (0.0, 90.0)  # still untouched


def test_update_cast_info_no_kwargs_is_a_no_op(tmp_path):
    path = tmp_path / "info.cops.dat"
    path.write_text(INFO_COPS_DAT)
    before = path.read_text()

    update_cast_info(path, "WISE_CAST_001_190817_220856_URC.csv")

    assert path.read_text() == before


def test_update_cast_info_creates_new_row_with_only_given_fields_set(tmp_path):
    path = tmp_path / "info.cops.dat"
    path.write_text(INFO_COPS_DAT)

    update_cast_info(
        path,
        "WISE_CAST_999_190817_235959_URC.csv",
        longitude=10.0,
        latitude=20.0,
        linear_max_delta_depth=[1.0, 2.0, 3.0, 4.0],
    )

    entries = read_info_cops(path)
    new_entry = next(e for e in entries if e.file == "WISE_CAST_999_190817_235959_URC.csv")
    assert new_entry.longitude == 10.0
    assert new_entry.latitude == 20.0
    assert new_entry.chl_flag is None  # padded to "NA", not left unset
    assert new_entry.linear_max_delta_depth == [1.0, 2.0, 3.0, 4.0]
    assert new_entry.time_window is None  # padded to "x"


ABSORPTION_COPS_DAT = """\
cops.file;320;340;380;443
WISE_CAST_001_190817_220856_URC.csv;9.2979;6.8733;3.8018;1.4911
WISE_CAST_002_190817_221224_URC.csv;9.2979;6.8733;3.8018;1.4911
"""


def test_read_absorption_cops(tmp_path):
    path = tmp_path / "absorption.cops.dat"
    path.write_text(ABSORPTION_COPS_DAT)

    table = read_absorption_cops(path)

    assert list(table.index) == ["WISE_CAST_001_190817_220856_URC.csv", "WISE_CAST_002_190817_221224_URC.csv"]
    waves, values = absorption_for_cast(table, "WISE_CAST_001_190817_220856_URC.csv")
    np.testing.assert_allclose(waves, [320.0, 340.0, 380.0, 443.0])
    np.testing.assert_allclose(values, [9.2979, 6.8733, 3.8018, 1.4911])
