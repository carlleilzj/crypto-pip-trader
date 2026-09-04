"""N-day breakout signal source with PIP-shape exit and hard stop.

Turtle-style entry: close breaks above the rolling N-bar high -> long;
breaks below the rolling N-bar low -> short (or flat in long_flat mode).
Keeps the PIP shape-break exit and hard stop that the diagnostics showed
are the strategy's core edge.

Vectorized ``breakout_pip_exit_signals`` and stateful ``FrozenBreakoutStrategy``
are written bar-for-bar identical — parity is locked by
tests/test_breakout_parity.py before any parameter search.
"""
from __future__ import annotations

import numpy as np

from research.momentum import shape_fit
from research.pips import find_pips


def breakout_pip_exit_signals(
    close: np.ndarray,
    n: int = 20,
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
    """N-bar high/low breakout entry, with PIP / stop / trail exits.

    A breakout at bar i means close[i] > max(close[i-n:i]) (long) or
    close[i] < min(close[i-n:i]) (short). The window is the n bars BEFORE i
    (not including i), so a new high at i is a genuine breakout.
    Exit-then-flat-next-bar semantics: exit bar is flat, re-entry next bar.
    """
    c = np.asarray(close, dtype=float)
    m = len(c)
    sig = np.zeros(m)
    if m < n + 1:
        return sig
    # Rolling max/min of the n bars before each index (exclusive of i).
    roll_high = np.array([c[i - n : i].max() if i >= n else np.nan for i in range(m)])
    roll_low = np.array([c[i - n : i].min() if i >= n else np.nan for i in range(m)])
    break_up = c > roll_high
    break_dn = c < roll_low

    i = n
    while i < m:
        if break_up[i]:
            side = 1.0
        elif break_dn[i]:
            side = -1.0 if mode == "long_short" else 0.0
        else:
            i += 1
            continue
        if side == 0.0:
            i += 1
            continue
        look = min(24, i)
        pred_w = c[i - look + 1 : i + 1]
        pred_y: list[float] | np.ndarray
        if len(pred_w) >= pip_n:
            _, pred_y = find_pips(pred_w, pip_n, 3)
        else:
            pred_y = pred_w
        end = min(i + hold, m)
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


class FrozenBreakoutStrategy:
    """Stateful N-bar breakout strategy, parity-identical to the vectorized path."""

    def __init__(
        self,
        n: int = 20,
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
        self.n = int(n)
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
        if i < self.n:
            return 0.0
        window = c[i - self.n : i]
        wh = float(window.max())
        wl = float(window.min())
        px = float(c[i])
        if px > wh:
            side = 1.0
        elif px < wl:
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
