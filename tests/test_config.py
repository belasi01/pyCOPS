from __future__ import annotations

import numpy as np
import pytest

from conftest import INFO_COPS_DAT, INIT_COPS_DAT
from pycops.io.config import absorption_for_cast, read_absorption_cops, read_info_cops, read_init_cops


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
    assert third.dark_files == ["dark_001.csv"]


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
