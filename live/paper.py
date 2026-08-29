from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from live.risk import RiskGuard
from research.costs import CostModel


def _oid(seq: int, bar: str, kind: str) -> str:
    safe = str(bar).replace(" ", "T").replace(":", "").replace("+", "")
    return f"PAPER-{seq:06d}-{kind}-{safe}"


@dataclass
class SimulatedOrder:
    client_order_id: str
    bar: str
    symbol: str
    side: str
    action: str
    reduce_only: bool
    qty: float
    raw_price: float
    fill_price: float
    fee: float
    slippage: float
    pnl: float
    status: str = "FILLED"
    ts: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaperBroker:
    cost: CostModel
    equity: float
    pos: int = 0
    qty: float = 0.0
    entry: float = 0.0
    symbol: str = "BTCUSDT"
    order_seq: int = 0
    last_orders: list[SimulatedOrder] = field(default_factory=list)

    def mark(self, price: float) -> float:
        if self.pos == 0:
            return self.equity
        return self.equity + self.pos * self.qty * (price - self.entry)

    def apply_funding(self, mark: float, rate: float) -> None:
        if self.pos == 0 or rate == 0:
            return
        pay = self.pos * self.qty * mark * rate
        self.equity -= pay

    def _close_position(self, price: float, bar: str) -> list[SimulatedOrder]:
        """Close the current position if any. Returns list of order(s)."""
        if self.pos == 0 or self.qty == 0:
            return []
        now = datetime.now(UTC).isoformat()
        fill = self.cost.fill_price(price, -self.pos)
        fee = abs(self.qty) * fill * self.cost.taker_fee
        pnl = self.pos * self.qty * (fill - self.entry)
        self.order_seq += 1
        close_side = "SELL" if self.pos > 0 else "BUY"
        order = SimulatedOrder(
            client_order_id=_oid(self.order_seq, bar, "CLOSE"),
            bar=bar,
            symbol=self.symbol,
            side=close_side,
            action="CLOSE",
            reduce_only=True,
            qty=abs(self.qty),
            raw_price=price,
            fill_price=fill,
            fee=fee,
            slippage=abs(fill - price) * abs(self.qty),
            pnl=pnl - fee,
            status="FILLED",
            ts=now,
        )
        self.equity += pnl - fee
        self.pos = 0
        self.qty = 0.0
        self.entry = 0.0
        return [order]

    def _open_position(self, side: int, price: float, risk: RiskGuard, bar: str) -> list[SimulatedOrder]:
        """Open a new position on the given side. Returns list of order(s)."""
        now = datetime.now(UTC).isoformat()
        mark_eq = self.equity  # after close, equity is current
        if not risk.check(mark_eq):
            return []
        fill = self.cost.fill_price(price, side)
        q = risk.target_qty(self.equity, fill)
        if q <= 0:
            return []
        fee = q * fill * self.cost.taker_fee
        self.order_seq += 1
        open_side = "BUY" if side > 0 else "SELL"
        order = SimulatedOrder(
            client_order_id=_oid(self.order_seq, bar, "OPEN"),
            bar=bar,
            symbol=self.symbol,
            side=open_side,
            action="OPEN",
            reduce_only=False,
            qty=q,
            raw_price=price,
            fill_price=fill,
            fee=fee,
            slippage=abs(fill - price) * q,
            pnl=0.0,
            status="FILLED",
            ts=now,
        )
        self.equity -= fee
        self.pos = side
        self.qty = q
        self.entry = fill
        return [order]

    def set_position(self, side: int, price: float, risk: RiskGuard, bar: str = "") -> list[SimulatedOrder]:
        """Set position to the given side (+, -, 0). Flattens first if needed."""
        self.last_orders = []
        orders = self._close_position(price, bar)
        if side != 0:
            orders += self._open_position(side, price, risk, bar)
        self.last_orders = orders
        return orders
