from __future__ import annotations

import numpy as np

from pycops.processing.ed0 import fit_ed0


def test_correction_compensates_cloud_dip():
    depth = np.linspace(0, 5, 200)
    waves = np.array([550.0])
    baseline = 100.0
    ed0_all = np.full((200, 1), baseline)
    dip_idx = 100
    ed0_all[dip_idx, 0] = baseline * 0.5  # a passing cloud briefly halves Ed0

    # QC'd subset excludes the dip (as tilt/depth QC would tend to smooth over
    # a brief instrument-level fluctuation less than removing it outright, but
    # for this test we simulate the fit being driven by the stable baseline).
    kept = np.ones(200, dtype=bool)
    kept[dip_idx] = False
    depth_grid = np.linspace(0, 5, 30)

    fit = fit_ed0(waves, depth[kept], ed0_all[kept], ed0_all, span=5.0, depth_grid=depth_grid, idx_depth_0=0)

    np.testing.assert_allclose(fit.value_at_0, [baseline], rtol=0.05)
    assert fit.correction[dip_idx, 0] > 1.5  # correction boosts the dipped scan back up
    np.testing.assert_allclose(fit.correction[0, 0], 1.0, rtol=0.05)


def test_correction_shape_matches_full_raw_matrix():
    depth = np.linspace(0, 5, 50)
    waves = np.array([440.0, 550.0])
    ed0_all = np.full((50, 2), 100.0)
    depth_grid = np.linspace(0, 5, 10)

    fit = fit_ed0(waves, depth, ed0_all, ed0_all, span=5.0, depth_grid=depth_grid)

    assert fit.correction.shape == ed0_all.shape


def test_smoothed_correction_does_not_amplify_a_single_scan_noise_spike():
    # The concrete case motivating the "smoothed" method: Simon found the raw correction (dividing
    # by each individual raw scan) injects too much of Ed0's own scan-to-scan noise into EdZ/LuZ/
    # EuZ. Same synthetic cloud-dip fixture as test_correction_compensates_cloud_dip above, where
    # correction_raw genuinely (and correctly) spikes at the dipped scan -- correction_smoothed
    # should stay near 1.0 there instead, since it's evaluated against the LOESS-smoothed Ed0 curve
    # (fit only from the surrounding, undipped scans) rather than the dipped raw value itself.
    depth = np.linspace(0, 5, 200)
    time = np.arange(200, dtype=float)  # elapsed seconds, 1 scan/sec -- monotonic by construction
    waves = np.array([550.0])
    baseline = 100.0
    ed0_all = np.full((200, 1), baseline)
    dip_idx = 100
    ed0_all[dip_idx, 0] = baseline * 0.5

    kept = np.ones(200, dtype=bool)
    kept[dip_idx] = False
    depth_grid = np.linspace(0, 5, 30)

    fit = fit_ed0(
        waves,
        depth[kept],
        ed0_all[kept],
        ed0_all,
        span=5.0,
        depth_grid=depth_grid,
        idx_depth_0=0,
        time_kept=time[kept],
        time_all=time,
    )

    assert fit.correction_raw[dip_idx, 0] > 1.5
    np.testing.assert_allclose(fit.correction_smoothed[dip_idx, 0], 1.0, rtol=0.05)


def test_method_selects_which_correction_is_active():
    depth = np.linspace(0, 5, 200)
    time = np.arange(200, dtype=float)
    waves = np.array([550.0])
    baseline = 100.0
    ed0_all = np.full((200, 1), baseline)
    ed0_all[100, 0] = baseline * 0.5
    kept = np.ones(200, dtype=bool)
    kept[100] = False
    depth_grid = np.linspace(0, 5, 30)

    fit_raw = fit_ed0(
        waves,
        depth[kept],
        ed0_all[kept],
        ed0_all,
        span=5.0,
        depth_grid=depth_grid,
        time_kept=time[kept],
        time_all=time,
        method="raw",
    )
    fit_smoothed = fit_ed0(
        waves,
        depth[kept],
        ed0_all[kept],
        ed0_all,
        span=5.0,
        depth_grid=depth_grid,
        time_kept=time[kept],
        time_all=time,
        method="smoothed",
    )

    assert fit_raw.method == "raw"
    np.testing.assert_array_equal(fit_raw.correction, fit_raw.correction_raw)
    assert fit_smoothed.method == "smoothed"
    np.testing.assert_array_equal(fit_smoothed.correction, fit_smoothed.correction_smoothed)
    # Both still compute both curves, regardless of which is "active" -- needed for the
    # diagnostic comparison plot.
    np.testing.assert_array_equal(fit_raw.correction_smoothed, fit_smoothed.correction_smoothed)
    np.testing.assert_array_equal(fit_raw.correction_raw, fit_smoothed.correction_raw)


def test_smoothed_correction_uses_time_not_depth_on_a_down_up_profile():
    """Regression test for a real bug found on real CASCADE station data (2026-08-08): depth is
    only "a monotonic proxy for time" when the cast actually is monotonic. A down-up profile puts
    two different real *times* -- possibly with genuinely different illumination -- at the *same*
    depth. An earlier version of correction_smoothed evaluated the depth-domain LOESS fit and
    interpolated it back onto each scan's own depth: at any depth visited by both legs, that
    blends together whatever Ed0 levels occurred on each leg, giving the *same* (wrong) smoothed
    value to both a bright descent scan and a dim ascent scan at that depth, regardless of when
    each one actually happened.

    Build a cast that descends 0->5 m under bright, stable illumination, then partially ascends
    5->2 m under dim, stable illumination (a cloud rolled in) -- so depths 2-5 m are revisited at
    two genuinely different brightness levels, while depth 0 (idx_depth_0) is only ever visited
    during the bright descent, keeping value_at_0 itself uncontaminated so this isolates
    correction_smoothed specifically."""
    n_down, n_up = 150, 80
    down_depth = np.linspace(0, 5, n_down)
    up_depth = np.linspace(5, 2, n_up)  # partial ascent, never returns to depth 0
    depth = np.concatenate([down_depth, up_depth])
    time = np.arange(n_down + n_up, dtype=float)

    waves = np.array([550.0])
    bright, dim = 100.0, 70.0
    ed0_all = np.concatenate([np.full(n_down, bright), np.full(n_up, dim)])[:, None]

    kept = np.ones(n_down + n_up, dtype=bool)
    depth_grid = np.linspace(0, 5, 30)

    # Two scans at essentially the same depth (~3.4 m), one from each leg -- the whole point of
    # this test is that they must NOT get the same correction, since they occurred at very
    # different times under very different real illumination.
    down_probe, up_probe = 100, n_down + 40

    fit = fit_ed0(
        waves,
        depth[kept],
        ed0_all[kept],
        ed0_all,
        span=1.0,
        depth_grid=depth_grid,
        idx_depth_0=0,
        time_kept=time[kept],
        time_all=time,
    )

    np.testing.assert_allclose(fit.value_at_0, [bright], rtol=0.01)  # uncontaminated by the dim leg
    # The bright descent scan's smoothed correction should track its own true (bright) level...
    np.testing.assert_allclose(fit.correction_smoothed[down_probe, 0], bright / bright, rtol=0.05)
    # ...while the dim ascent scan's should track its own true (dim) level -- not the same,
    # depth-blended value the two would get if depth were (wrongly) used as the time axis.
    np.testing.assert_allclose(fit.correction_smoothed[up_probe, 0], bright / dim, rtol=0.05)


def test_smoothed_correction_clamps_rather_than_extrapolates_beyond_time_window():
    """Regression test for a second real bug found on real CASCADE station data (2026-08-08),
    surfacing right after the depth-vs-time fix above: a cast with a saved ``time.window`` (see
    ``depth.py``'s ``time_window_mask``) trims the *training* set (``time_kept``/``ed0_kept``) to
    a sub-range of the cast, but correction_smoothed must still be evaluated at *every* scan
    (``time_all``), including ones well outside that trained window (e.g. a pre-descent bobbing
    phase before the window even starts). Querying a degree-2 local polynomial fit tens of
    seconds beyond its own training range can overshoot wildly -- confirmed on a real cast as
    correction_smoothed flipping *negative* right at the very start. Every query point must be
    clamped into the training data's own time range (interpolation only, never extrapolation),
    matching aop_cleaning.py's own "clamp, don't extrapolate" convention elsewhere in this port."""
    waves = np.array([550.0])
    baseline = 100.0

    # Training data only covers t in [100, 200] (simulating a time.window that excludes the
    # cast's early scans); Ed0 has a real, mild upward trend within that window.
    time_kept = np.linspace(100.0, 200.0, 100)
    ed0_kept = (baseline + 0.1 * (time_kept - 100.0))[:, None]

    # Query points span the *whole* cast, including far outside the trained window.
    time_all = np.linspace(0.0, 224.0, 300)
    ed0_all = np.full((300, 1), baseline)

    depth_kept = time_kept  # depth_range/idx_depth_0 machinery needs *a* depth axis; irrelevant here
    depth_grid = np.linspace(0.0, 5.0, 20)

    fit = fit_ed0(
        waves,
        depth_kept,
        ed0_kept,
        ed0_all,
        span=100.0,  # wide relative to depth_kept's own tiny range -> a wide, extrapolation-prone fraction
        depth_grid=depth_grid,
        idx_depth_0=0,
        time_kept=time_kept,
        time_all=time_all,
    )

    assert np.all(np.isfinite(fit.correction_smoothed))
    assert np.all(fit.correction_smoothed > 0)  # never flips sign from wild extrapolation
    # Every out-of-window query holds at the boundary's own trained value, not extrapolated
    # further away from it.
    before_window = time_all < 100.0
    after_window = time_all > 200.0
    at_start_of_window = fit.correction_smoothed[np.argmin(np.abs(time_all - 100.0)), 0]
    at_end_of_window = fit.correction_smoothed[np.argmin(np.abs(time_all - 200.0)), 0]
    np.testing.assert_allclose(fit.correction_smoothed[before_window, 0], at_start_of_window, rtol=1e-6)
    np.testing.assert_allclose(fit.correction_smoothed[after_window, 0], at_end_of_window, rtol=1e-6)
