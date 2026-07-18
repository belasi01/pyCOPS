from __future__ import annotations

import numpy as np
import pytest

from pycops.processing.detection_limits import detection_limit_for_waves


def test_matches_table_at_known_wavelength():
    limit = detection_limit_for_waves("LuZ", np.array([443.0]))
    np.testing.assert_allclose(limit, [4.00e-05])


def test_matches_table_for_edz_and_euz():
    edz = detection_limit_for_waves("EdZ", np.array([305.0]))
    euz = detection_limit_for_waves("EuZ", np.array([305.0]))
    np.testing.assert_allclose(edz, [4.00e-03])
    np.testing.assert_allclose(euz, [4.00e-03])


def test_interpolates_between_table_points():
    limit = detection_limit_for_waves("LuZ", np.array([400.0]))
    # between the 380 nm (2e-4) and 395 nm (5e-5) table entries
    assert 0 < limit[0] < 2e-4


def test_unknown_instrument_raises():
    with pytest.raises(KeyError):
        detection_limit_for_waves("Ed0", np.array([443.0]))


def test_returns_array_matching_input_shape():
    waves = np.array([320.0, 443.0, 665.0, 875.0])
    limit = detection_limit_for_waves("LuZ", waves)
    assert limit.shape == waves.shape
