import numpy as np
import pandas as pd

from backtest.engine import EventBacktest
from research.costs import CostModel


def test_run_signals_with_sizes():
    n = 80
    rng = np.random.default_rng(2)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open_time": idx,
            "open": open_,
            "high": np.maximum(open_, close),
            "low": np.minimum(open_, close),
            "close": close,
            "log_close": np.log(close),
            "funding_rate": 0.0,
        }
    )
    sig = np.ones(n)
    sig[:10] = 0
    sizes = np.full(n, 0.3)
    bt = EventBacktest(CostModel(taker_fee=0.0004, slippage_bps=1.0), start_equity=1000.0, skip_after_losing_fold=False)
    res = bt.run_signals(df, sig, sizes)
    assert len(res.equity) == n
    assert res.summary["n_trades"] >= 1


def test_drawdown_halt_flattens():
    n = 120
    close = np.concatenate([np.full(20, 100.0), np.linspace(100, 50, n - 20)])
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open_time": idx,
            "open": open_,
            "high": np.maximum(open_, close),
            "low": np.minimum(open_, close),
            "close": close,
            "log_close": np.log(close),
            "funding_rate": 0.0,
        }
    )
    sig = np.ones(n)
    bt = EventBacktest(
        CostModel(taker_fee=0.0, slippage_bps=0.0, use_funding=False),
        start_equity=1000.0,
        notional_frac=1.0,
        skip_after_losing_fold=False,
        max_drawdown_halt=0.15,
    )
    res = bt.run_signals(df, sig)
    assert res.summary["halted"] is True
    assert res.summary["max_dd"] > -0.25
