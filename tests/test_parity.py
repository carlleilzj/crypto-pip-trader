"""Parity test: vectorized research signal must equal the stateful live signal.

This is the regression that previously slipped through: the vectorized
``ts_momentum_pip_exit`` could reverse direction (long->short) on the same
bar it exits, while ``FrozenMomentumStrategy.on_bar_close`` returns 0.0 on
the exit bar and only re-enters the next bar. The two paths therefore
produced different signals (~91.6% agreement, ~5-7x trade-count gap),
which meant the gate metrics were computed on a signal the live bot never
trades. This test pins them equal across all exit modes.
"""
from __future__ import annotations

import numpy as np
import pytest

from live.strategy import FrozenMomentumStrategy
from research.momentum import ts_momentum_pip_exit, ts_momentum_signals


def _synthetic(seed: int = 1, n: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, n)))


def _assert_parity(close: np.ndarray, *, pip: bool, stop_pct: float | None,
                    mode: str = "long_short") -> None:
    kwargs = dict(lookback=720, hold=720, mode=mode, pip_n=5, min_bars=24, fit_exit=0.0)
    sig_bt = ts_momentum_pip_exit(close, use_pip=pip, stop_pct=stop_pct, **kwargs)
    strat = FrozenMomentumStrategy(exit_mode="pip" if pip else "time", stop_pct=stop_pct, **kwargs)
    sig_live = np.array([strat.on_bar_close(close[: i + 1]) for i in range(len(close))])
    mism = int(np.sum(sig_bt != sig_live))
    assert mism == 0, f"{mism} mismatches (pip={pip}, stop={stop_pct})"


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_parity_pip_plus_stop_synthetic(seed: int) -> None:
    """Frozen spec (pip exit + 6% stop) on synthetic walks."""
    _assert_parity(_synthetic(seed), pip=True, stop_pct=0.06)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_parity_time_only_synthetic(seed: int) -> None:
    """Pure time exit (no pip, no stop) must also match."""
    _assert_parity(_synthetic(seed), pip=False, stop_pct=None)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_parity_stop_only_synthetic(seed: int) -> None:
    """Stop-only exit (no pip) must also match."""
    _assert_parity(_synthetic(seed), pip=False, stop_pct=0.08)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_parity_pip_only_synthetic(seed: int) -> None:
    """PIP-only exit (no stop) must also match."""
    _assert_parity(_synthetic(seed), pip=True, stop_pct=None)


def test_parity_baseline_ts_momentum_signals() -> None:
    """The plain time-momentum baseline must match a time-exit live strategy."""
    close = _synthetic(3)
    sig_bt = ts_momentum_signals(close, lookback=720, hold=720, mode="long_short")
    strat = FrozenMomentumStrategy(
        lookback=720, hold=720, mode="long_short", exit_mode="time", stop_pct=None,
    )
    sig_live = np.array([strat.on_bar_close(close[: i + 1]) for i in range(len(close))])
    mism = int(np.sum(sig_bt != sig_live))
    assert mism == 0, f"baseline mismatch: {mism} bars"


def test_no_direct_reversal_in_vectorized() -> None:
    """The vectorized path must never flip long->short on adjacent bars.

    This is the structural guarantee the stateful strategy gives for free
    (exit returns 0.0, re-entry is next bar); the vectorized path must match.
    """
    for seed in range(20):
        close = _synthetic(seed)
        sig = ts_momentum_pip_exit(
            close, lookback=720, hold=720, mode="long_short",
            pip_n=5, min_bars=24, fit_exit=0.0, use_pip=True, stop_pct=0.06,
        )
        d = np.diff(sig)
        reversals = int(np.sum((d != 0) & (sig[1:] == -sig[:-1]) & (sig[:-1] != 0)))
        assert reversals == 0, f"seed {seed}: {reversals} direct reversals"


def test_parity_on_real_btc_oos() -> None:
    """End-to-end parity on the real BTC OOS slice the gate runs on."""
    from pathlib import Path

    from research.data import load_dataset
    from research.momentum import oos_slice

    root = Path(__file__).resolve().parents[1]
    kp, fp = root / "data/BTCUSDT_1h.csv", root / "data/BTCUSDT_funding.csv"
    if not kp.exists() or not fp.exists():
        pytest.skip("BTC data not present")
    df = load_dataset(kp, fp)
    close = oos_slice(df, 17520)["close"].to_numpy(dtype=float)
    _assert_parity(close, pip=True, stop_pct=0.06)
