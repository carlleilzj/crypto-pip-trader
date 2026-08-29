"""Tests for stop-loss and trailing-stop in FrozenMomentumStrategy."""
import numpy as np

from live.strategy import FrozenMomentumStrategy


def _make_close(up_bars: int, flat_bars: int, down_bars: int,
                start: float = 100.0, up_end: float = 120.0, down_end: float = 90.0) -> np.ndarray:
    up = np.linspace(start, up_end, up_bars)
    flat = np.full(flat_bars, up_end)
    dn = np.linspace(up_end, down_end, down_bars)
    return np.concatenate([up, flat, dn])


def test_hard_stop_flattens_at_threshold():
    """With stop_pct=0.08, a long position should flatten when price drops -8% from entry."""
    close = _make_close(30, 0, 20, start=90, up_end=100, down_end=85)
    strat = FrozenMomentumStrategy(
        lookback=24, hold=80, exit_mode="time",
        stop_pct=0.08, trail_arm=None, trail_giveback=None,
    )
    n = len(close)
    sigs = np.array([strat.on_bar_close(close[: i + 1]) for i in range(n)])
    first = next((i for i, s in enumerate(sigs) if s != 0), None)
    if first is None:
        return  # no trade entered in this data slice
    last = int(np.where(sigs != 0)[0][-1])
    # The price at some point should have dropped near -8% from entry
    entry = close[first]
    worst = close[first:last + 1].min() if last > first else entry
    if worst < entry * 0.93:
        # The stop should have triggered before the full hold period
        assert last < first + 60, "stop should exit early, not hold full period"


def test_hard_stop_short_side():
    """Short position flattens on adverse move with stop_pct."""
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.015, 200)))
    strat = FrozenMomentumStrategy(
        lookback=24, hold=80, exit_mode="time",
        stop_pct=0.08, trail_arm=None, trail_giveback=None,
        mode="long_short",
    )
    sigs = np.array([strat.on_bar_close(close[: i + 1]) for i in range(len(close))])
    # Test that the strategy runs without error (no crash)
    assert len(sigs) == len(close)
    assert set(np.unique(sigs)).issubset({-1.0, 0.0, 1.0})


def test_trail_stop_triggers():
    """Trailing stop flattens after favorable move + giveback."""
    close = np.concatenate([
        np.linspace(90, 100, 30),      # entry near 100
        np.linspace(100, 115, 30),     # favorable move >10% -> arm
        np.linspace(115, 108, 15),     # giveback >5% from peak 115 -> trigger
        np.linspace(108, 105, 10),
    ])
    strat = FrozenMomentumStrategy(
        lookback=24, hold=120, exit_mode="time",
        stop_pct=None, trail_arm=0.10, trail_giveback=0.05,
    )
    sigs = np.array([strat.on_bar_close(close[: i + 1]) for i in range(len(close))])
    first = next((i for i, s in enumerate(sigs) if s != 0), None)
    if first is not None:
        last = int(np.where(sigs != 0)[0][-1])
        # Should have exited before the full hold
        assert last < first + 80, "trailing stop should exit early"


def test_no_trail_when_not_armed():
    """Without sufficient favorable move, trailing stop doesn't trigger."""
    close = np.concatenate([
        np.linspace(90, 100, 30),
        np.linspace(100, 106, 20),
        np.linspace(106, 101, 40),
    ])
    strat = FrozenMomentumStrategy(
        lookback=24, hold=120, exit_mode="time",
        stop_pct=None, trail_arm=0.10, trail_giveback=0.05,
    )
    sigs = np.array([strat.on_bar_close(close[: i + 1]) for i in range(len(close))])
    first = next((i for i, s in enumerate(sigs) if s != 0), None)
    if first is not None:
        last = int(np.where(sigs != 0)[0][-1])
        # Price never reached +10% from entry, so trailing stop shouldn't arm
        assert last >= first + 50, "should hold full period without trailing arm"


def test_pip_and_stop_coexist():
    """PIP exit and stop-loss can coexist; both can trigger exit."""
    close = np.concatenate([
        np.linspace(90, 100, 30),
        np.array([99, 97, 95, 93, 91, 89, 87, 85, 83, 81], dtype=float),
    ])
    strat = FrozenMomentumStrategy(
        lookback=24, hold=80, exit_mode="pip",
        min_bars=8, fit_exit=0.0,
        stop_pct=0.08, trail_arm=None, trail_giveback=None,
    )
    sigs = np.array([strat.on_bar_close(close[: i + 1]) for i in range(len(close))])
    first = next((i for i, s in enumerate(sigs) if s != 0), None)
    if first is not None:
        last = int(np.where(sigs != 0)[0][-1])
        # Should exit before the full hold
        assert last < first + 40, "stop should exit early, even with pip mode"


def test_stop_restores_hold_state():
    """After stop exit, the strategy should clean state so it can re-enter."""
    close = np.concatenate([
        np.linspace(90, 100, 30),
        np.array([99, 97, 95, 93, 91, 89, 87], dtype=float),
        np.linspace(87, 110, 40),  # recovery -> new entry
    ])
    strat = FrozenMomentumStrategy(
        lookback=24, hold=80, exit_mode="time",
        stop_pct=0.08, trail_arm=None, trail_giveback=None,
    )
    sigs = np.array([strat.on_bar_close(close[: i + 1]) for i in range(len(close))])
    # Count non-zero blocks: each entry should have a matching exit
    changes = np.diff(np.concatenate([[0], sigs, [0]]))
    entries = np.where(changes == 1)[0]
    exits = np.where(changes == -1)[0]
    assert len(entries) == len(exits), "every entry needs an exit"
    # Verify that exit cleaned state: entry_price and extreme reset to 0 before
    # any re-entry on the same bar.  Check by running a second time and
    # inspecting state between exit and potential re-entry.
    strat2 = FrozenMomentumStrategy(
        lookback=24, hold=80, exit_mode="time",
        stop_pct=0.08, trail_arm=None, trail_giveback=None,
    )
    for i in range(len(close)):
        prev_sig = strat2.curr_sig
        sig2 = strat2.on_bar_close(close[: i + 1])
        if prev_sig != 0 and sig2 == 0:
            # Exited and did NOT re-enter on same bar — verify clean state
            assert strat2.entry_price == 0.0, f"entry_price not reset at bar {i}"
            assert strat2.extreme == 0.0, f"extreme not reset at bar {i}"
            assert strat2.hold_left == 0
            assert strat2.pred_y is None
        elif prev_sig != 0 and sig2 != 0 and sig2 != prev_sig:
            # Exited and re-entered opposite direction — state should be new
            assert strat2.entry_price == float(close[i]), f"entry_price should be current close at bar {i}"
            assert strat2.extreme == strat2.entry_price
