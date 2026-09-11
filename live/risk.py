from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskGuard:
    notional_frac: float = 0.5
    max_leverage: float = 2.0
    daily_loss_halt: float = 0.05
    max_drawdown_halt: float = 0.15
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""

    def reset_day(self, equity: float) -> None:
        self.day_start_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def check(self, equity: float) -> bool:
        if self.halted:
            # A daily-loss halt releases at the next UTC day (runloop resets
            # day_start_equity and clears halt state on rollover); a
            # max-drawdown halt stays until the operator clears it, because
            # the drawdown condition does not expire with the day.
            if self.halt_reason.startswith("max_dd"):
                return False
            self.halted = False
            self.halt_reason = ""
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity > 0:
            dd = 1.0 - equity / self.peak_equity
            if dd >= self.max_drawdown_halt:
                self.halted = True
                self.halt_reason = f"max_dd {dd:.3f}"
                return False
        if self.day_start_equity > 0:
            day_loss = 1.0 - equity / self.day_start_equity
            if day_loss >= self.daily_loss_halt:
                self.halted = True
                self.halt_reason = f"daily_loss {day_loss:.3f}"
                return False
        return True

    def target_qty(self, equity: float, price: float) -> float:
        if price <= 0 or equity <= 0:
            return 0.0
        notional = equity * min(self.notional_frac, self.max_leverage)
        return notional / price
