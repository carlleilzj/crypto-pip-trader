import numpy as np
import pandas as pd

from live.paper import PaperBroker
from live.risk import RiskGuard
from live.strategy import FrozenMomentumStrategy
from research.costs import CostModel


def test_momentum_time_exit_holds_block():
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    strat = FrozenMomentumStrategy(lookback=48, hold=48, mode="long_short", exit_mode="time")
    live = [strat.on_bar_close(close[: i + 1]) for i in range(len(close))]
    first = next(i for i, s in enumerate(live) if s != 0)
    held = live[first : first + 48]
    assert all(s == live[first] for s in held)


def test_pip_exit_can_flatten():
    up = np.linspace(100, 140, 80)
    dn = np.linspace(140, 90, 80)
    close = np.concatenate([up, dn])
    strat = FrozenMomentumStrategy(lookback=24, hold=80, mode="long_short", exit_mode="pip", min_bars=8)
    sigs = [strat.on_bar_close(close[: i + 1]) for i in range(len(close))]
    timed = FrozenMomentumStrategy(lookback=24, hold=80, mode="long_short", exit_mode="time")
    tsigs = [timed.on_bar_close(close[: i + 1]) for i in range(len(close))]
    assert sum(1 for s in sigs if s != 0) <= sum(1 for s in tsigs if s != 0)


def test_replay_from_matches_bar_by_bar():
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    live = FrozenMomentumStrategy(lookback=24, hold=40, mode="long_short", exit_mode="pip", min_bars=8)
    sigs = [live.on_bar_close(close[: i + 1]) for i in range(len(close))]
    batch = FrozenMomentumStrategy(lookback=24, hold=40, mode="long_short", exit_mode="pip", min_bars=8)
    last = batch.replay_from(close, 0)
    assert last == sigs[-1]
    assert batch.hold_left == live.hold_left
    assert batch.bars_in_trade == live.bars_in_trade
    assert batch.curr_sig == live.curr_sig


def test_replay_from_catches_up_after_gap():
    up = np.linspace(100, 140, 80)
    dn = np.linspace(140, 90, 80)
    close = np.concatenate([up, dn])
    live = FrozenMomentumStrategy(lookback=24, hold=80, mode="long_short", exit_mode="pip", min_bars=8)
    for i in range(50):
        live.on_bar_close(close[: i + 1])
    paused = FrozenMomentumStrategy(lookback=24, hold=80, mode="long_short", exit_mode="pip", min_bars=8)
    paused.hold_left = live.hold_left
    paused.curr_sig = live.curr_sig
    paused.bars_in_trade = live.bars_in_trade
    paused.pred_y = list(live.pred_y) if live.pred_y is not None else None
    for i in range(50, len(close)):
        live.on_bar_close(close[: i + 1])
    paused.replay_from(close, 50)
    assert paused.curr_sig == live.curr_sig
    assert paused.bars_in_trade == live.bars_in_trade
    assert paused.hold_left == live.hold_left


def test_fetch_public_spot_klines_closed():
    from research.data import fetch_public_spot_klines

    df = fetch_public_spot_klines(limit=5, closed_only=True)
    assert len(df) >= 3
    assert "close" in df.columns
    assert df["open_time"].is_monotonic_increasing
    now = pd.Timestamp.now(tz="UTC")
    if "close_time" in df.columns:
        assert df["close_time"].max() < now


def test_paper_broker_roundtrip():
    cost = CostModel(taker_fee=0.0004, slippage_bps=1.0)
    broker = PaperBroker(cost=cost, equity=1000.0)
    risk = RiskGuard(notional_frac=0.5, daily_loss_halt=0.9, max_drawdown_halt=0.9)
    risk.reset_day(1000.0)
    opens = broker.set_position(1, 100.0, risk, bar="2026-01-01 00:00:00+00:00")
    assert broker.pos == 1
    assert len(opens) == 1
    assert opens[0].action == "OPEN"
    assert opens[0].status == "FILLED"
    assert opens[0].side == "BUY"
    closes = broker.set_position(0, 101.0, risk, bar="2026-01-01 01:00:00+00:00")
    assert broker.pos == 0
    assert len(closes) == 1
    assert closes[0].action == "CLOSE"
    assert closes[0].reduce_only is True
    assert broker.equity != 1000.0
