from __future__ import annotations

from conftest import INFO_COPS_DAT, INIT_COPS_DAT
from pycops.io.config import read_info_cops, read_init_cops


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
