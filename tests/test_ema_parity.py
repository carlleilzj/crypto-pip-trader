"""Parity test for the EMA-crossover signal: vectorized == stateful.

This is the safeguard that the 30d-momentum strategy lacked — the two paths
drifted to ~91.6% agreement and nobody noticed for weeks. Written first,
before any parameter search, so a parity break fails the build immediately.
"""
from __future__ import annotations

import numpy as np
import pytest

from live.strategy import FrozenMomentumStrategy  # noqa: F401  (sanity import)
from research.ema import FrozenEMAStrategy, ema_pip_exit_signals


def _synthetic(seed: int = 1, n: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, n)))


def _assert_parity(close, *, fast, slow, use_pip, stop_pct, mode="long_short"):
    sig_bt = ema_pip_exit_signals(
        close, fast=fast, slow=slow, hold=720, mode=mode,
        pip_n=5, min_bars=24, fit_exit=0.0, use_pip=use_pip, stop_pct=stop_pct,
    )
    s = FrozenEMAStrategy(
        fast=fast, slow=slow, hold=720, mode=mode, pip_n=5, min_bars=24,
        fit_exit=0.0, use_pip=use_pip, stop_pct=stop_pct,
    )
    sig_live = np.array([s.on_bar_close(close[: i + 1]) for i in range(len(close))])
    mism = int(np.sum(sig_bt != sig_live))
    assert mism == 0, f"{mism} mismatches (fast={fast} slow={slow} pip={use_pip} stop={stop_pct})"


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_parity_pip_plus_stop(seed):
    _assert_parity(_synthetic(seed), fast=50, slow=200, use_pip=True, stop_pct=0.06)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_parity_stop_only(seed):
    _assert_parity(_synthetic(seed), fast=50, slow=200, use_pip=False, stop_pct=0.08)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_parity_time_only(seed):
    _assert_parity(_synthetic(seed), fast=20, slow=100, use_pip=False, stop_pct=None)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_parity_pip_only(seed):
    _assert_parity(_synthetic(seed), fast=50, slow=200, use_pip=True, stop_pct=None)


def test_no_direct_reversal_in_vectorized():
    """EMA crossover must never flip long->short on adjacent bars."""
    for seed in range(20):
        close = _synthetic(seed)
        sig = ema_pip_exit_signals(
            close, fast=50, slow=200, hold=720, mode="long_short",
            pip_n=5, min_bars=24, fit_exit=0.0, use_pip=True, stop_pct=0.06,
        )
        d = np.diff(sig)
        rev = int(np.sum((d != 0) & (sig[1:] == -sig[:-1]) & (sig[:-1] != 0)))
        assert rev == 0, f"seed {seed}: {rev} direct reversals"
