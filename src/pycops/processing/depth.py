"""Depth QC and depth-binning, ported from ``derived.data.R`` (the ``Depth.good`` /
``depth.fitted`` construction) and ``process.LuZ.R`` / ``process.EdZ.R`` (binning inputs).
"""

from __future__ import annotations

import numpy as np

from pycops.processing.filters import median_filter


def good_depth_mask(depth: np.ndarray, k: int = 3) -> np.ndarray:
    """Flag scans whose depth is a sensor glitch, via a wide-tolerance median filter.

    The tolerance scales with the profile (``max(depth) / n * 50``, as in the R
    package) so it stays permissive for normal depth noise and only catches
    genuine outliers.
    """
    depth = np.asarray(depth, dtype=float)
    delta = np.max(depth) / len(depth) * 50
    filtered = median_filter(depth, k=k, delta=delta, fill=True, replace=False)
    return ~np.isnan(filtered)


def depth_grid(discretization: list[float], max_depth: float | None = None) -> np.ndarray:
    """Build the adaptive depth-binning grid described by ``depth.discretization``.

    ``discretization`` is a flat ``[start, step, next_start, step, next_start, ...]``
    sequence (as in ``init.cops.dat``): each ``(start, step, next_start)`` triplet
    produces evenly spaced points from ``start`` up to (but not including)
    ``next_start``, so resolution can coarsen with depth. If given, ``max_depth``
    truncates the grid to the deepest usable scan.
    """
    d = np.asarray(discretization, dtype=float)
    segments = []
    for i in range(0, len(d) - 2, 2):
        start, step, next_start = d[i], d[i + 1], d[i + 2]
        stop = next_start - step / 2
        n = int(np.floor((stop - start) / step + 1e-9)) + 1
        n = max(n, 0)
        segments.append(start + step * np.arange(n))

    grid = np.concatenate(segments) if segments else np.array([], dtype=float)
    if max_depth is not None:
        grid = grid[grid <= max_depth]
    return grid


def _bin_edges(grid: np.ndarray) -> np.ndarray:
    mid = (grid[:-1] + grid[1:]) / 2
    first_edge = grid[0] - (mid[0] - grid[0])
    last_edge = grid[-1] + (grid[-1] - mid[-1])
    return np.concatenate([[first_edge], mid, [last_edge]])


def bin_by_depth(
    depth: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average ``values`` into the depth bins centered on ``grid``.

    ``values`` is ``(n_scans,)`` or ``(n_scans, n_channels)``; bin edges are the
    midpoints between consecutive ``grid`` points. NaNs in ``values`` and scans
    excluded by ``mask`` (e.g. a tilt or depth-QC mask) are ignored. Returns
    ``(binned, counts)`` where ``binned`` has NaN for empty bins.
    """
    depth = np.asarray(depth, dtype=float)
    values = np.asarray(values, dtype=float)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    if mask is None:
        mask = np.ones(depth.shape, dtype=bool)

    edges = _bin_edges(grid)
    bin_idx = np.searchsorted(edges, depth, side="right") - 1
    n_bins = len(grid)
    in_range = mask & (bin_idx >= 0) & (bin_idx < n_bins)

    counts = np.bincount(bin_idx[in_range], minlength=n_bins)[:n_bins]
    binned = np.full((n_bins, values.shape[1]), np.nan)
    for j in range(values.shape[1]):
        col = values[:, j]
        finite = in_range & np.isfinite(col)
        sums = np.bincount(bin_idx[finite], weights=col[finite], minlength=n_bins)[:n_bins]
        n = np.bincount(bin_idx[finite], minlength=n_bins)[:n_bins]
        with np.errstate(invalid="ignore"):
            binned[:, j] = np.where(n > 0, sums / np.maximum(n, 1), np.nan)

    if squeeze:
        binned = binned[:, 0]
    return binned, counts
