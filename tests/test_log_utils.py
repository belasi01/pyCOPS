from __future__ import annotations

import logging

from pycops.processing.log_utils import station_log_handler


def test_station_log_handler_writes_info_messages(tmp_path):
    logger = logging.getLogger("pycops.processing.deployment")
    with station_log_handler(tmp_path / "nc"):
        logger.info("hello")

    log_path = tmp_path / "nc" / "processing.log"
    assert log_path.exists()
    assert "hello" in log_path.read_text()


def test_station_log_handler_creates_nc_dir_if_missing(tmp_path):
    nc_dir = tmp_path / "nc"
    assert not nc_dir.exists()

    with station_log_handler(nc_dir):
        pass

    assert nc_dir.is_dir()


def test_station_log_handler_default_mode_truncates(tmp_path):
    logger = logging.getLogger("pycops.processing.deployment")
    with station_log_handler(tmp_path / "nc"):
        logger.info("first run")
    with station_log_handler(tmp_path / "nc"):
        logger.info("second run")

    text = (tmp_path / "nc" / "processing.log").read_text()
    assert "first run" not in text
    assert "second run" in text


def test_station_log_handler_append_mode_preserves_history(tmp_path):
    logger = logging.getLogger("pycops.processing.deployment")
    with station_log_handler(tmp_path / "nc"):
        logger.info("first run")
    with station_log_handler(tmp_path / "nc", mode="a"):
        logger.info("second run")

    text = (tmp_path / "nc" / "processing.log").read_text()
    assert "first run" in text
    assert "second run" in text


def test_station_log_handler_detaches_after_exit(tmp_path):
    pycops_logger = logging.getLogger("pycops")
    handlers_before = list(pycops_logger.handlers)

    with station_log_handler(tmp_path / "nc"):
        assert len(pycops_logger.handlers) == len(handlers_before) + 1

    assert pycops_logger.handlers == handlers_before


def test_station_log_handler_detaches_even_on_exception(tmp_path):
    pycops_logger = logging.getLogger("pycops")
    handlers_before = list(pycops_logger.handlers)

    try:
        with station_log_handler(tmp_path / "nc"):
            raise ValueError("boom")
    except ValueError:
        pass

    assert pycops_logger.handlers == handlers_before
