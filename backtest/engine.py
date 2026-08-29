from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from live.risk import RiskGuard
from research.costs import CostModel
from research.metrics import martin_ratio, max_drawdown, summarize_trades


@dataclass
class Trade:
    entry_i: int
    exit_i: int
    side: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    fee: float
    funding: float
    slippage: float


@dataclass
class BacktestResult:
    equity: np.ndarray
    signals: np.ndarray
    trades: pd.DataFrame
    summary: dict
    folds: list = field(default_factory=list)


class EventBacktest:
    def __init__(
        self,
        cost: CostModel,
        start_equity: float = 1000.0,
        notional_frac: float = 0.5,
        skip_after_losing_fold: bool = True,
        max_drawdown_halt: float | None = None,
    ):
        self.cost = cost
        self.start_equity = start_equity
        self.notional_frac = notional_frac
        self.skip_after_losing_fold = skip_after_losing_fold
        self.max_drawdown_halt = max_drawdown_halt

    def run_signals(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        sizes: np.ndarray | None = None,
    ) -> BacktestResult:
        n = len(df)
        if len(signals) != n:
            raise ValueError("signals length must match bars")
        open_px = df["open"].to_numpy(dtype=float)
        close_px = df["close"].to_numpy(dtype=float)
        funding = (
            df["funding_rate"].to_numpy(dtype=float)
            if "funding_rate" in df.columns
            else np.zeros(n)
        )
        signals = np.asarray(signals, dtype=float)
        if sizes is None:
            sizes = np.full(n, self.notional_frac, dtype=float)
        else:
            sizes = np.asarray(sizes, dtype=float)
            if len(sizes) != n:
                raise ValueError("sizes length must match bars")
        equity = np.full(n, self.start_equity, dtype=float)
        cash = self.start_equity
        pos = 0
        qty = 0.0
        entry_price = 0.0
        entry_i = -1
        fee_acc = 0.0
        fund_acc = 0.0
        slip_acc = 0.0
        pending = 0
        pending_size = self.notional_frac
        current_size = 0.0
        trades: list[Trade] = []
        risk = None
        if self.max_drawdown_halt is not None:
            risk = RiskGuard(
                notional_frac=self.notional_frac,
                daily_loss_halt=1.0,
                max_drawdown_halt=self.max_drawdown_halt,
            )
            risk.reset_day(self.start_equity)
            risk.peak_equity = self.start_equity

        def close_position(i: int, px: float) -> None:
            nonlocal cash, pos, qty, entry_price, entry_i, fee_acc, fund_acc, slip_acc, current_size
            if pos == 0 or qty == 0:
                return
            fill = self.cost.fill_price(px, -pos)
            fee = abs(qty) * fill * self.cost.taker_fee
            pnl = pos * qty * (fill - entry_price)
            cash += pnl - fee
            trades.append(
                Trade(
                    entry_i=entry_i,
                    exit_i=i,
                    side=pos,
                    entry_price=entry_price,
                    exit_price=fill,
                    qty=qty,
                    pnl=pnl - fee,
                    fee=fee_acc + fee,
                    funding=fund_acc,
                    slippage=slip_acc + abs(fill - px) * qty,
                )
            )
            pos = 0
            qty = 0.0
            entry_price = 0.0
            entry_i = -1
            fee_acc = 0.0
            fund_acc = 0.0
            slip_acc = 0.0
            current_size = 0.0

        def open_position(i: int, side: int, px: float, size: float) -> None:
            nonlocal cash, pos, qty, entry_price, entry_i, fee_acc, slip_acc, current_size
            fill = self.cost.fill_price(px, side)
            if fill <= 0 or cash <= 0:
                return
            q = (cash * max(0.0, size)) / fill
            if q <= 0:
                return
            fee = q * fill * self.cost.taker_fee
            cash -= fee
            pos = side
            qty = q
            entry_price = fill
            entry_i = i
            fee_acc = fee
            slip_acc = abs(fill - px) * q
            current_size = size

        for i in range(n):
            if pos != 0 and self.cost.use_funding and funding[i] != 0.0:
                pay = pos * qty * open_px[i] * funding[i]
                cash -= pay
                fund_acc += pay

            need_rebalance = pending != pos or (
                pending != 0 and abs(pending_size - current_size) > 0.05
            )
            if need_rebalance:
                if pos != 0:
                    close_position(i, open_px[i])
                if pending != 0:
                    open_position(i, pending, open_px[i], pending_size)

            pending = int(np.sign(signals[i]))
            pending_size = float(sizes[i])
            if risk is not None and risk.halted:
                pending = 0
                pending_size = 0.0

            mtm = cash
            if pos != 0:
                mtm += pos * qty * (close_px[i] - entry_price)
            equity[i] = mtm
            if risk is not None and not risk.check(mtm):
                pending = 0
                pending_size = 0.0

        if pos != 0:
            close_position(n - 1, close_px[-1])
            equity[-1] = cash

        tdf = pd.DataFrame([t.__dict__ for t in trades])
        if tdf.empty:
            tdf = pd.DataFrame(
                columns=[
                    "entry_i",
                    "exit_i",
                    "side",
                    "entry_price",
                    "exit_price",
                    "qty",
                    "pnl",
                    "fee",
                    "funding",
                    "slippage",
                ]
            )

        log_eq = np.log(np.maximum(equity, 1e-12))
        log_rets = np.diff(log_eq, prepend=log_eq[0])
        summary = summarize_trades(tdf, equity)
        summary.update(
            {
                "martin": martin_ratio(log_rets),
                "max_dd": max_drawdown(equity),
                "fees": float(tdf["fee"].sum()) if len(tdf) else 0.0,
                "funding": float(tdf["funding"].sum()) if len(tdf) else 0.0,
                "start_equity": self.start_equity,
                "return": float(equity[-1] / self.start_equity - 1.0) if n else 0.0,
                "halted": bool(risk.halted) if risk is not None else False,
                "halt_reason": risk.halt_reason if risk is not None else "",
            }
        )
        return BacktestResult(
            equity=equity,
            signals=signals,
            trades=tdf,
            summary=summary,
            folds=[],
        )

