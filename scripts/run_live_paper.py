#!/usr/bin/env python3
"""Public-market paper step. No API key, no orders.

Uses Binance public spot 1H klines (data-api.binance.vision) because USDT-M
REST is geo-blocked (HTTP 451). Signal is the frozen 30d momentum spec.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live.paper import PaperBroker  # noqa: E402
from live.risk import RiskGuard  # noqa: E402
from live.strategy import FrozenMomentumStrategy  # noqa: E402
from research.config import PassedConfig  # noqa: E402
from research.costs import CostModel  # noqa: E402
from research.data import fetch_public_spot_klines  # noqa: E402
from research.log import get_logger  # noqa: E402

log = get_logger("live_paper")

STATE_PATH = ROOT / "reports" / "live_paper_state.json"
LOG_PATH = ROOT / "reports" / "live_paper_log.csv"
ORDERS_PATH = ROOT / "reports" / "live_paper_orders.csv"


def load_state(path: Path, start_equity: float) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "equity": start_equity,
        "pos": 0,
        "qty": 0.0,
        "entry": 0.0,
        "hold_left": 0,
        "curr_sig": 0.0,
        "peak_equity": start_equity,
        "day": None,
        "day_start_equity": start_equity,
        "halted": False,
        "halt_reason": "",
        "last_bar": None,
        "n_steps": 0,
        "n_trades": 0,
        "n_orders": 0,
        "order_seq": 0,
        "source": None,
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=float))


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def append_log(row: dict) -> None:
    append_csv(LOG_PATH, [row])


def main() -> None:
    cfg = PassedConfig.load(ROOT / "configs" / "passed.yaml")
    frac = cfg.notional_frac
    start_eq = cfg.backtest.start_equity

    bars_needed = cfg.lookback + 2
    klines = fetch_public_spot_klines(
        symbol=cfg.symbol,
        interval=cfg.interval,
        limit=bars_needed + 5,
        closed_only=True,
    )
    if len(klines) <= cfg.lookback:
        raise SystemExit(f"not enough closed bars: {len(klines)}")

    last = klines.iloc[-1]
    bar_id = str(last["open_time"])
    state = load_state(STATE_PATH, start_eq)
    if state.get("last_bar") == bar_id:
        log.info("already processed bar %s", bar_id)
        return

    cost = CostModel(
        taker_fee=cfg.costs.taker_fee,
        slippage_bps=cfg.costs.slippage_bps,
        use_funding=False,
    )
    broker = PaperBroker(
        cost=cost,
        equity=float(state["equity"]),
        pos=int(state["pos"]),
        qty=float(state["qty"]),
        entry=float(state["entry"]),
        symbol=cfg.symbol,
        order_seq=int(state.get("order_seq") or 0),
    )
    risk = RiskGuard(
        notional_frac=frac,
        max_leverage=cfg.risk.max_leverage,
        daily_loss_halt=cfg.risk.daily_loss_halt,
        max_drawdown_halt=cfg.risk.max_drawdown_halt,
        day_start_equity=float(state["day_start_equity"]),
        peak_equity=float(state["peak_equity"]),
        halted=bool(state.get("halted")),
        halt_reason=str(state.get("halt_reason") or ""),
    )
    strat = FrozenMomentumStrategy(
        lookback=cfg.lookback,
        hold=cfg.hold,
        mode=cfg.mode,
        exit_mode=cfg.exit_mode,
        pip_n=cfg.pip_n,
        min_bars=cfg.min_bars,
        fit_exit=cfg.fit_exit,
        stop_pct=cfg.stop_pct,
        trail_arm=cfg.trail_arm,
        trail_giveback=cfg.trail_giveback,
    )

    # Replay all klines through strategy
    closes = klines["close"].to_numpy(dtype=float)
    n = len(closes)
    sig = 0.0
    for i in range(n):
        sig = strat.on_bar_close(closes[: i + 1])

    mark_px = float(closes[-1])
    day = bar_id[:10] if bar_id else None
    if state.get("day") != day:
        risk.reset_day(broker.mark(mark_px))
        state["day"] = day

    side = 1 if sig > 0 else -1 if sig < 0 else 0
    orders = []
    if side == 0 and broker.pos == 0 or side == broker.pos:
        pass
    else:
        orders = broker.set_position(side, mark_px, risk, bar=bar_id)
    traded = 1 if orders else 0

    state.clear()
    state.update(
        equity=broker.equity,
        pos=broker.pos,
        qty=broker.qty,
        entry=broker.entry,
        hold_left=strat.hold_left,
        curr_sig=float(sig),
        bars_in_trade=strat.bars_in_trade,
        last_fit=strat.last_fit,
        pred_y=strat.pred_y,
        extreme=strat.extreme,
        peak_equity=risk.peak_equity,
        day=day,
        day_start_equity=risk.day_start_equity,
        halted=risk.halted,
        halt_reason=risk.halt_reason,
        last_bar=bar_id,
        n_steps=int(state.get("n_steps") or 0) + 1,
        n_trades=int(state.get("n_trades") or 0) + traded,
        n_orders=int(state.get("n_orders") or 0) + len(orders),
        order_seq=broker.order_seq,
        source="spot_public",
        note="simulated FILLED orders only; no exchange API",
    )
    save_state(STATE_PATH, state)
    append_log(
        {
            "bar": bar_id,
            "close": mark_px,
            "signal": sig,
            "pos": broker.pos,
            "equity": state["equity"],
            "halted": risk.halted,
            "n_orders": len(orders),
            "source": "spot_public",
        }
    )
    append_csv(ORDERS_PATH, [o.to_dict() for o in orders])
    log.info(
        "bar=%s sig=%d pos=%d equity=%.2f orders=%d",
        bar_id, sig, broker.pos, state["equity"], len(orders),
    )


if __name__ == "__main__":
    main()
