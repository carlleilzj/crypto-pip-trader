"""EMA crossover signal source with PIP-shape exit and hard stop.

Replaces the 30d time-momentum entry with an EMA(fast)/EMA(slow) crossover,
keeping the PIP shape-break exit and hard stop that the diagnostic showed
are the strategy's core edge (pure time-momentum: BTC +0.3% train; with
PIP+stop: +77%).

The vectorized ``ema_pip_exit_signals`` and the stateful ``FrozenEMAStrategy``
are written to be bar-for-bar identical — the parity regression that let the
30d-momentum gate run on a signal the live bot never trades must not recur.
See tests/test_ema_parity.py.
"""
from __future__ import annotations

import numpy as np

from research.momentum import shape_fit
from research.pips import find_pips


def _ema(close: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average. span>=1; span=1 == price itself."""
    c = np.asarray(close, dtype=float)
    if span <= 1:
        return c.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(c)
    out[0] = c[0]
    for i in range(1, len(c)):
        out[i] = alpha * c[i] + (1.0 - alpha) * out[i - 1]
    return out


def ema_pip_exit_signals(
    close: np.ndarray,
    fast: int = 50,
    slow: int = 200,
    hold: int = 720,
    mode: str = "long_short",
    pip_n: int = 5,
    min_bars: int = 24,
    fit_exit: float = 0.0,
    use_pip: bool = True,
    stop_pct: float | None = None,
    trail_arm: float | None = None,
    trail_giveback: float | None = None,
) -> np.ndarray:
    """EMA(fast)/EMA(slow) crossover entry, with PIP-shape / stop / trail exits.

    Entry: fast crosses above slow -> long; below -> short (or flat in
    long_flat mode). Holds for ``hold`` bars unless an exit fires first.
    Exit-then-flat-next-bar semantics: the exit bar is flat, re-entry is the
    bar after — matching ``FrozenEMAStrategy.on_bar_close`` exactly.
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    sig = np.zeros(n)
    if n < 2:
        return sig
    ef = _ema(c, fast)
    es = _ema(c, slow)
    above = ef > es
    # A crossover at bar i means the relation flipped between i-1 and i.
    cross_up = np.zeros(n, dtype=bool)
    cross_dn = np.zeros(n, dtype=bool)
    cross_up[1:] = above[1:] & ~above[:-1]
    cross_dn[1:] = ~above[1:] & above[:-1]

    # Start scanning from bar 1 — EMA is defined from bar 0, so a crossover
    # at bar 1 is valid (though noisy in the warm-up period). This matches
    # the stateful strategy, which processes bars from the start and would
    # enter on any crossover. (Skipping i<slow here was the parity break.)
    i = 1
    while i < n:
        if cross_up[i]:
            side = 1.0
        elif cross_dn[i]:
            side = -1.0 if mode == "long_short" else 0.0
        else:
            i += 1
            continue
        if side == 0.0:
            i += 1
            continue
        # Freeze the PIP prediction shape from the entry window (last min(24,..) bars).
        look = min(24, i)
        pred_w = c[i - look + 1 : i + 1]
        pred_y: list[float] | np.ndarray
        if len(pred_w) >= pip_n:
            _, pred_y = find_pips(pred_w, pip_n, 3)
        else:
            pred_y = pred_w
        end = min(i + hold, n)
        exit_i = end
        entry = float(c[i])
        extreme = entry
        for j in range(i + 1, end):
            px = float(c[j])
            if side > 0:
                extreme = max(extreme, px)
                pnl = px / entry - 1.0
                dd = 1.0 - px / extreme if extreme > 0 else 0.0
            else:
                extreme = min(extreme, px)
                pnl = 1.0 - px / entry
                dd = px / extreme - 1.0 if extreme > 0 else 0.0
            if stop_pct is not None and pnl <= -stop_pct:
                exit_i = j
                break
            if trail_arm is not None and trail_giveback is not None and pnl >= trail_arm and dd >= trail_giveback:
                exit_i = j
                break
            if use_pip and (j - i) >= min_bars and (j - i - min_bars) % 12 == 0:
                actual_w = c[i : j + 1]
                k = min(pip_n, len(actual_w))
                if k >= 3:
                    _, actual_y = find_pips(actual_w, k, 3)
                    if shape_fit(np.asarray(pred_y), np.asarray(actual_y)) < fit_exit:
                        exit_i = j
                        break
        sig[i:exit_i] = side
        i = exit_i + 1 if exit_i > i else i + 1
    return sig


class FrozenEMAStrategy:
    """Stateful EMA-crossover strategy, parity-identical to ema_pip_exit_signals.

    Mirrors FrozenMomentumStrategy's structure so the live runloop can swap it
    in with minimal change: same on_bar_close / replay_from contract.
    """

    def __init__(
        self,
        fast: int = 50,
        slow: int = 200,
        hold: int = 720,
        mode: str = "long_short",
        pip_n: int = 5,
        min_bars: int = 24,
        fit_exit: float = 0.0,
        use_pip: bool = True,
        stop_pct: float | None = None,
        trail_arm: float | None = None,
        trail_giveback: float | None = None,
    ):
        self.fast = int(fast)
        self.slow = int(slow)
        self.hold = int(hold)
        self.mode = mode
        self.pip_n = pip_n
        self.min_bars = min_bars
        self.fit_exit = fit_exit
        self.use_pip = use_pip
        self.stop_pct = stop_pct
        self.trail_arm = trail_arm
        self.trail_giveback = trail_giveback
        self.hold_left = 0
        self.bars_in_trade = 0
        self.curr_sig = 0.0
        self.pred_y: list[float] | None = None
        self.last_fit: float | None = None
        self.entry_price = 0.0
        self.extreme = 0.0
        self._ef: np.ndarray | None = None
        self._es: np.ndarray | None = None
        self._above: np.ndarray | None = None
        self._last_n = 0

    def _ensure_emas(self, c: np.ndarray) -> None:
        n = len(c)
        if self._ef is not None and self._last_n >= n:
            return
        self._ef = _ema(c, self.fast)
        self._es = _ema(c, self.slow)
        self._above = self._ef > self._es
        self._last_n = n

    def _cross_up(self, i: int) -> bool:
        if i < 1 or self._above is None or i >= len(self._above):
            return False
        return bool(self._above[i]) and not bool(self._above[i - 1])

    def _cross_dn(self, i: int) -> bool:
        if i < 1 or self._above is None or i >= len(self._above):
            return False
        return (not bool(self._above[i])) and bool(self._above[i - 1])

    def _freeze_pred(self, c: np.ndarray, i: int) -> None:
        look = min(24, i + 1)
        pred_w = c[i - look + 1 : i + 1]
        if len(pred_w) >= self.pip_n:
            _, py = find_pips(pred_w, self.pip_n, 3)
            self.pred_y = [float(x) for x in py]
        else:
            self.pred_y = [float(x) for x in pred_w]

    def _shape_broken(self, c: np.ndarray, i: int) -> bool:
        if self.pred_y is None or self.bars_in_trade < self.min_bars:
            return False
        if (self.bars_in_trade - self.min_bars) % 12 != 0:
            return False
        window = c[i - self.bars_in_trade : i + 1]
        k = min(self.pip_n, len(window))
        if k < 3:
            return False
        _, actual_y = find_pips(window, k, 3)
        fit = shape_fit(np.asarray(self.pred_y), np.asarray(actual_y))
        self.last_fit = fit
        return fit < self.fit_exit

    def _stop_triggered(self, px: float) -> bool:
        entry = self.entry_price
        if entry == 0.0:
            return False
        if self.curr_sig > 0:
            self.extreme = max(self.extreme, px)
            pnl = px / entry - 1.0
            dd = 1.0 - px / self.extreme if self.extreme > 0 else 0.0
        else:
            self.extreme = min(self.extreme, px)
            pnl = 1.0 - px / entry
            dd = px / self.extreme - 1.0 if self.extreme > 0 else 0.0
        if self.stop_pct is not None and pnl <= -self.stop_pct:
            return True
        return (
            self.trail_arm is not None
            and self.trail_giveback is not None
            and pnl >= self.trail_arm
            and dd >= self.trail_giveback
        )

    def on_bar_close(self, close_window: np.ndarray) -> float:
        c = np.asarray(close_window, dtype=float)
        self._ensure_emas(c)
        i = len(c) - 1
        if self.curr_sig != 0.0:
            self.hold_left = max(0, self.hold_left - 1)
            self.bars_in_trade += 1
            time_done = self.hold_left == 0
            shape_done = self.use_pip and self._shape_broken(c, i)
            stop_done = self._stop_triggered(float(c[i]))
            if not (time_done or shape_done or stop_done):
                return self.curr_sig
            self.curr_sig = 0.0
            self.hold_left = 0
            self.bars_in_trade = 0
            self.pred_y = None
            self.entry_price = 0.0
            self.extreme = 0.0
            return 0.0
        # Entry: only on a fresh crossover this bar.
        if self._cross_up(i):
            side = 1.0
        elif self._cross_dn(i):
            side = -1.0 if self.mode == "long_short" else 0.0
        else:
            return 0.0
        if side == 0.0:
            return 0.0
        self.curr_sig = side
        self.hold_left = self.hold
        self.bars_in_trade = 0
        self.last_fit = None
        self.entry_price = float(c[i])
        self.extreme = self.entry_price
        self._freeze_pred(c, i)
        return self.curr_sig

    def replay_from(self, closes: np.ndarray, start_idx: int) -> float:
        c = np.asarray(closes, dtype=float)
        sig = float(self.curr_sig)
        n = len(c)
        start = max(0, int(start_idx))
        for i in range(start, n):
            sig = self.on_bar_close(c[: i + 1])
        return sig
