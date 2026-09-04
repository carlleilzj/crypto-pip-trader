from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from research.momentum import shape_fit
from research.pips import find_pips

if TYPE_CHECKING:
    from research.config import PassedConfig


def make_strategy(config: PassedConfig) -> Any:
    """Factory: pick the strategy class from config.signal.

    All frozen strategies share the same on_bar_close / replay_from contract,
    so RunLoop can swap them without other changes. ``signal`` in passed.yaml
    selects: "momentum" (30d), "ema" (EMA crossover), "breakout" (N-bar).
    """
    common = dict(
        hold=int(config.hold),
        mode=str(config.mode),
        pip_n=int(config.pip_n),
        min_bars=int(config.min_bars),
        fit_exit=float(config.fit_exit),
        use_pip=bool(config.exit_mode == "pip"),
        stop_pct=(float(config.stop_pct) if config.stop_pct is not None else None),
        trail_arm=(float(config.trail_arm) if config.trail_arm is not None else None),
        trail_giveback=(float(config.trail_giveback) if config.trail_giveback is not None else None),
    )  # type: dict[str, Any]
    sig = (config.signal or "momentum").lower()
    if sig == "breakout":
        from research.breakout import FrozenBreakoutStrategy
        return FrozenBreakoutStrategy(n=int(config.n), **common)
    if sig == "ema":
        from research.ema import FrozenEMAStrategy
        return FrozenEMAStrategy(fast=int(config.fast), slow=int(config.slow), **common)
    # default: 30d time-momentum (legacy spec, parity-fixed)
    return FrozenMomentumStrategy(
        lookback=config.lookback,
        exit_mode=config.exit_mode,  # momentum uses exit_mode, others use use_pip
        **{k: v for k, v in common.items() if k != "use_pip"},
    )


class FrozenMomentumStrategy:
    """30-day time-series momentum with PIP shape exit, hard stop, and trailing stop.

    Entry: sign of 720h (30d) past return.
    Exit modes:
      - time: hold for exactly ``hold`` bars then flatten.
      - pip: flatten early when shape fit (z-scored cosine) drops below ``fit_exit``.
      - stop_pct: flatten if adverse move from entry exceeds this fraction.
      - trail_arm/trail_giveback: flatten after a favorable move >= trail_arm
        if price gives back trail_giveback from the in-trade extreme.
    """

    def __init__(
        self,
        lookback: int = 720,
        hold: int = 720,
        mode: str = "long_short",
        exit_mode: str = "time",
        pip_n: int = 5,
        min_bars: int = 24,
        fit_exit: float = 0.0,
        stop_pct: float | None = None,
        trail_arm: float | None = None,
        trail_giveback: float | None = None,
    ):
        self.lookback = int(lookback)
        self.hold = int(hold)
        self.mode = mode
        self.exit_mode = exit_mode
        self.pip_n = pip_n
        self.min_bars = min_bars
        self.fit_exit = fit_exit
        self.stop_pct = stop_pct
        self.trail_arm = trail_arm
        self.trail_giveback = trail_giveback
        self.hold_left = 0
        self.bars_in_trade = 0
        self.curr_sig = 0.0
        self.pred_y: list[float] | None = None
        self.last_fit: float | None = None
        self.entry_price: float = 0.0
        self.extreme: float = 0.0

    def _entry_side(self, c: np.ndarray) -> float:
        if len(c) <= self.lookback:
            return 0.0
        past = c[-1] / c[-1 - self.lookback] - 1.0
        if past > 0:
            return 1.0
        if self.mode == "long_flat":
            return 0.0
        return -1.0

    def _freeze_pred(self, c: np.ndarray) -> None:
        look = min(24, len(c))
        if look >= self.pip_n:
            _, py = find_pips(c[-look:], self.pip_n, 3)
            self.pred_y = [float(x) for x in py]
        else:
            self.pred_y = [float(x) for x in c[-look:]]

    def _shape_broken(self, c: np.ndarray) -> bool:
        if self.pred_y is None or self.bars_in_trade < self.min_bars:
            return False
        if (self.bars_in_trade - self.min_bars) % 12 != 0:
            return False
        window = c[-(self.bars_in_trade + 1):]
        k = min(self.pip_n, len(window))
        if k < 3:
            return False
        _, actual_y = find_pips(window, k, 3)
        fit = shape_fit(np.asarray(self.pred_y), np.asarray(actual_y))
        self.last_fit = fit
        return fit < self.fit_exit

    def _stop_triggered(self, px: float) -> bool:
        """Check hard stop-loss and trailing stop against current price.

        Updates ``self.extreme`` as a side effect so the extreme is always
        current for the next bar.
        """
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
        # Trailing stop: after favorable move >= trail_arm, flatten on giveback
        return (
            self.trail_arm is not None
            and self.trail_giveback is not None
            and pnl >= self.trail_arm
            and dd >= self.trail_giveback
        )

    def on_bar_close(self, close_window: np.ndarray) -> float:
        c = np.asarray(close_window, dtype=float)
        if self.curr_sig != 0.0:
            self.hold_left = max(0, self.hold_left - 1)
            self.bars_in_trade += 1
            time_done = self.hold_left == 0
            shape_done = self.exit_mode == "pip" and self._shape_broken(c)
            stop_done = self._stop_triggered(float(c[-1]))
            if not (time_done or shape_done or stop_done):
                return self.curr_sig
            self.curr_sig = 0.0
            self.hold_left = 0
            self.bars_in_trade = 0
            self.pred_y = None
            self.entry_price = 0.0
            self.extreme = 0.0
            return 0.0
        side = self._entry_side(c)
        if side == 0.0:
            return 0.0
        self.curr_sig = side
        self.hold_left = self.hold
        self.bars_in_trade = 0
        self.last_fit = None
        self.entry_price = float(c[-1])
        self.extreme = self.entry_price
        self._freeze_pred(c)
        return self.curr_sig

    def replay_from(self, closes: np.ndarray, start_idx: int) -> float:
        """Apply on_bar_close for bars [start_idx, end). Returns last signal."""
        c = np.asarray(closes, dtype=float)
        sig = float(self.curr_sig)
        n = len(c)
        start = max(0, int(start_idx))
        for i in range(start, n):
            end = i + 1
            if end <= self.lookback:
                continue
            sig = self.on_bar_close(c[:end])
        return sig
