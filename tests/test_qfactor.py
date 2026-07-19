from __future__ import annotations

import numpy as np
import pytest

from pycops.processing.qfactor import compute_q_factor


def test_compute_q_factor_none_chl_is_pi():
    q = compute_q_factor(None, 4)
    np.testing.assert_allclose(q, np.pi)
    assert q.shape == (4,)


def test_compute_q_factor_nan_chl_is_pi():
    q = compute_q_factor(float("nan"), 3)
    np.testing.assert_allclose(q, np.pi)


def test_compute_q_factor_chl_999_is_pi():
    q = compute_q_factor(999.0, 5)
    np.testing.assert_allclose(q, np.pi)


def test_compute_q_factor_chl_zero_is_pi():
    q = compute_q_factor(0.0, 2)
    np.testing.assert_allclose(q, np.pi)


def test_compute_q_factor_positive_chl_not_implemented():
    with pytest.raises(NotImplementedError):
        compute_q_factor(2.5, 4)
