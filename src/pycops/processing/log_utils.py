"""Per-station processing log.

Simon's R workflow printed extensive step-by-step progress to the console during processing;
pycops persists an equivalent -- major steps and full tracebacks on failure, not every
intermediate filter count (that level of detail is deliberately out of scope here, see
:func:`station_log_handler`'s docstring) -- to a ``processing.log`` file next to each station's
own ``nc/`` output. Motivated by a real case (Amundsen 2026, station CS1-9): a cast failed with
only a bare ``ValueError: `x` must contain at least 2 elements.`` surfaced to the UI, giving no
clue which processing step raised it or why -- the full traceback now lands in this log file,
right next to the very ``nc/`` folder someone would already be looking at to investigate.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_LOGGER_NAME = "pycops"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


@contextmanager
def station_log_handler(nc_dir: Path, mode: str = "w") -> Iterator[None]:
    """Attach a ``FileHandler`` writing to ``nc_dir / "processing.log"`` for the duration of the
    ``with`` block, then detach and close it -- avoids leaking handlers across repeated calls in
    a long-lived process (e.g. Streamlit's own server, which keeps running across button clicks).

    ``mode="w"`` (default) truncates -- a fresh full-deployment run (:func:`process_deployment`)
    starts a clean log. ``mode="a"`` appends -- a single-cast reprocess
    (:func:`reprocess_single_cast`) shouldn't wipe out the rest of the station's own log history
    just because one cast was re-run.

    Every ``pycops.*`` module logger (``logging.getLogger(__name__)``) propagates up to this
    handler automatically -- no per-module wiring needed. Kept at INFO level (major steps: cast
    started, instruments fit, shadow correction applied/skipped and why, Rrs source/method, QWIP
    pass/fail, bottom reflectance, and any exception with its full traceback) rather than R's own
    much chattier per-filter verbosity, to stay readable for the failure-diagnosis use case that
    motivated this rather than reproducing every intermediate count.
    """
    nc_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(nc_dir / "processing.log", mode=mode, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.setLevel(logging.INFO)

    logger = logging.getLogger(_LOGGER_NAME)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        handler.close()
