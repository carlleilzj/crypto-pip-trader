import numpy as np

from research.momentum import (
    block_permute_close,
    buy_hold_signals,
    calibrate_notional,
    shape_fit,
    ts_momentum_pip_exit,
    ts_momentum_signals,
    vol_target_sizes,
)


def test_ts_momentum_hold_blocks():
    close = np.concatenate([np.linspace(100, 120, 200), np.linspace(120, 90, 200)])
    sig = ts_momentum_signals(close, lookback=24, hold=24)
    assert sig[:24].sum() == 0
    assert set(np.unique(sig)).issubset({-1.0, 0.0, 1.0})
    changes = np.where(np.diff(sig) != 0)[0]
    if len(changes) >= 2:
        assert (changes[1] - changes[0]) >= 23


def test_buy_hold():
    s = buy_hold_signals(10)
    assert len(s) == 10
    assert (s == 1).all()


def test_vol_target_bounds():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    sizes = vol_target_sizes(close, target_vol=0.40, vol_lookback=48, min_size=0.1, max_size=1.0)
    assert len(sizes) == 500
    assert sizes.min() >= 0.1 - 1e-9
    assert sizes.max() <= 1.0 + 1e-9


def test_vol_target_frozen_on_signal():
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    sig = ts_momentum_signals(close, lookback=24, hold=48)
    sizes = vol_target_sizes(close, vol_lookback=24, signals=sig)
    for i in range(1, len(sig)):
        if sig[i] == sig[i - 1] and sig[i] != 0:
            assert sizes[i] == sizes[i - 1]


def test_block_permute_length():
    close = np.linspace(100, 120, 500)
    out = block_permute_close(close, block=24, seed=0)
    assert len(out) == len(close)
    assert np.isclose(out[0], close[0])


def test_long_flat_never_short():
    close = np.concatenate([np.linspace(100, 80, 80), np.linspace(80, 120, 80)])
    sig = ts_momentum_signals(close, lookback=24, hold=24, mode="long_flat")
    assert (sig >= 0).all()


def test_calibrate_notional_scales_down():
    assert calibrate_notional(-0.40, 0.5, target_dd=0.20) == 0.25
    assert calibrate_notional(-0.10, 0.5, target_dd=0.20) == 0.5


def test_shape_fit_identical_is_one():
    y = np.array([1.0, 2.0, 3.0, 2.5, 4.0])
    assert shape_fit(y, y) > 0.99


def test_pip_exit_can_flatten_before_hold():
    up = np.linspace(100, 140, 80)
    dn = np.linspace(140, 90, 80)
    close = np.concatenate([up, dn])
    raw = ts_momentum_signals(close, lookback=24, hold=80)
    pip = ts_momentum_pip_exit(close, lookback=24, hold=80, min_bars=8, fit_exit=0.0)
    held_raw = int(np.sum(raw != 0))
    held_pip = int(np.sum(pip != 0))
    assert held_pip <= held_raw


def test_hard_stop_flattens_from_known_entry():
    close = np.concatenate([np.linspace(90, 100, 30), np.array([99, 98, 97, 96, 95, 94, 93, 92, 91, 90], dtype=float)])
    sig = ts_momentum_pip_exit(
        close, lookback=24, hold=80, use_pip=False, stop_pct=0.08, trail_arm=None, trail_giveback=None
    )
    first = next(i for i, s in enumerate(sig) if s != 0)
    last = int(np.where(sig != 0)[0][-1])
    entry = close[first]
    assert close[last] <= entry * 0.93


def test_trail_giveback_from_known_entry():
    close = np.concatenate([np.linspace(90, 100, 30), np.linspace(100, 112, 20), np.linspace(112, 105, 20)])
    sig = ts_momentum_pip_exit(
        close, lookback=24, hold=80, use_pip=False, stop_pct=None, trail_arm=0.10, trail_giveback=0.05
    )
    first = next(i for i, s in enumerate(sig) if s != 0)
    last = int(np.where(sig != 0)[0][-1])
    peak = close[first : last + 1].max()
    assert peak >= close[first] * 1.10
    assert close[last] <= peak * 0.96


def test_trail_not_armed_below_10pct():
    close = np.concatenate([np.linspace(90, 100, 30), np.linspace(100, 106, 20), np.linspace(106, 101, 40)])
    sig = ts_momentum_pip_exit(
        close, lookback=24, hold=80, use_pip=False, stop_pct=None, trail_arm=0.10, trail_giveback=0.05
    )
    first = next(i for i, s in enumerate(sig) if s != 0)
    last = int(np.where(sig != 0)[0][-1])
    assert close[first : last + 1].max() < close[first] * 1.10
    assert last >= first + 50
