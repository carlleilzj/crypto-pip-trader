"""Parity test for the N-bar breakout signal: vectorized == stateful."""
from __future__ import annotations

import numpy as np
import pytest

from research.breakout import FrozenBreakoutStrategy, breakout_pip_exit_signals


def _synthetic(seed: int = 1, n: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, n)))


def _assert_parity(close, *, n, use_pip, stop_pct, mode="long_short"):
    sig_bt = breakout_pip_exit_signals(
        close, n=n, hold=720, mode=mode, pip_n=5, min_bars=24,
        fit_exit=0.0, use_pip=use_pip, stop_pct=stop_pct,
    )
    s = FrozenBreakoutStrategy(
        n=n, hold=720, mode=mode, pip_n=5, min_bars=24, fit_exit=0.0,
        use_pip=use_pip, stop_pct=stop_pct,
    )
    sig_live = np.array([s.on_bar_close(close[: i + 1]) for i in range(len(close))])
    mism = int(np.sum(sig_bt != sig_live))
    assert mism == 0, f"{mism} mismatches (n={n} pip={use_pip} stop={stop_pct})"


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_parity_pip_plus_stop(seed):
    _assert_parity(_synthetic(seed), n=20, use_pip=True, stop_pct=0.06)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_parity_stop_only(seed):
    _assert_parity(_synthetic(seed), n=20, use_pip=False, stop_pct=0.08)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_parity_time_only(seed):
    _assert_parity(_synthetic(seed), n=55, use_pip=False, stop_pct=None)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_parity_pip_only(seed):
    _assert_parity(_synthetic(seed), n=20, use_pip=True, stop_pct=None)


def test_no_direct_reversal_in_vectorized():
    for seed in range(20):
        close = _synthetic(seed)
        sig = breakout_pip_exit_signals(
            close, n=20, hold=720, mode="long_short", pip_n=5, min_bars=24,
            fit_exit=0.0, use_pip=True, stop_pct=0.06,
        )
        d = np.diff(sig)
        rev = int(np.sum((d != 0) & (sig[1:] == -sig[:-1]) & (sig[:-1] != 0)))
        assert rev == 0, f"seed {seed}: {rev} direct reversals"
