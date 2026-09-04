#!/usr/bin/env python3
"""Public-market paper trading for the frozen 3-coin breakout portfolio.

No API key, no orders. Pulls public spot 1H klines from
data-api.binance.vision (USDT-M REST is geo-blocked HTTP 451), runs the
frozen breakout strategy bar-by-bar, and records simulated FILLED orders
with slippage + fees. State persists between runs so cron can call this
once per closed bar.

Single-coin legacy mode kept for backward compat (the old state file is
migrated). The 3-coin portfolio is the frozen spec since 2026-08-31.

Usage:
  python scripts/run_live_paper.py            # process latest closed bar (cron)
  python scripts/run_live_paper.py --loop 60  # long-running, poll every 60s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live.paper import PaperBroker  # noqa: E402
from live.risk import RiskGuard  # noqa: E402
from live.strategy import make_strategy  # noqa: E402
from research.config import PassedConfig  # noqa: E402
from research.costs import CostModel  # noqa: E402
from research.data import fetch_public_spot_klines  # noqa: E402
from research.log import get_logger  # noqa: E402

log = get_logger("live_paper")

STATE_PATH = ROOT / "reports" / "live_paper_state.json"
LOG_PATH = ROOT / "reports" / "live_paper_log.csv"
ORDERS_PATH = ROOT / "reports" / "live_paper_orders.csv"


def _default_state(start_equity: float) -> dict:
    return {
        "equity": start_equity,
        "per_coin": {},
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
        "source": "spot_public_3coin",
    }


def _default_coin_state() -> dict:
    return {
        "pos": 0,
        "qty": 0.0,
        "entry": 0.0,
        "hold_left": 0,
        "curr_sig": 0.0,
        "bars_in_trade": 0,
        "last_fit": None,
        "pred_y": None,
        "extreme": 0.0,
        "order_seq": 0,
    }


def load_state(path: Path, start_equity: float) -> dict:
    if path.exists():
        st = json.loads(path.read_text())
        # migrate legacy single-coin state -> 3-coin portfolio
        if "per_coin" not in st:
            log.info("migrating legacy single-coin state to 3-coin portfolio")
            old = {
                "pos": int(st.get("pos") or 0),
                "qty": float(st.get("qty") or 0),
                "entry": float(st.get("entry") or 0),
                "hold_left": int(st.get("hold_left") or 0),
                "curr_sig": float(st.get("curr_sig") or 0),
                "bars_in_trade": int(st.get("bars_in_trade") or 0),
                "last_fit": st.get("last_fit"),
                "pred_y": st.get("pred_y"),
                "extreme": float(st.get("extreme") or 0),
                "order_seq": int(st.get("order_seq") or 0),
            }
            st = _default_state(start_equity)
            st["equity"] = float(st.get("equity") or start_equity)
            st["per_coin"]["BTCUSDT"] = old
        return st
    return _default_state(start_equity)


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=float))
    tmp.replace(path)


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def step_once(cfg: PassedConfig) -> dict:
    """Process the latest closed bar for all coins. Returns summary dict."""
    symbols = cfg.symbols
    n_sym = len(symbols)
    frac_each = cfg.notional_frac / n_sym
    sub_capital = cfg.backtest.start_equity / n_sym
    # breakout needs n+2 bars min; pull generous history for replay accuracy.
    bars_needed = max(cfg.n, cfg.lookback, 50) + cfg.hold + 10

    state = load_state(STATE_PATH, cfg.backtest.start_equity)
    total_equity = float(state.get("equity") or cfg.backtest.start_equity)
    order_seq = int(state.get("order_seq") or 0)
    prev_n_steps = int(state.get("n_steps") or 0)
    prev_n_trades = int(state.get("n_trades") or 0)
    prev_n_orders = int(state.get("n_orders") or 0)
    prev_last_bar = state.get("last_bar")

    cost = CostModel(
        taker_fee=cfg.costs.taker_fee,
        slippage_bps=cfg.costs.slippage_bps,
        use_funding=False,  # spot public, no funding
    )
    risk = RiskGuard(
        notional_frac=frac_each,
        max_leverage=cfg.risk.max_leverage,
        daily_loss_halt=cfg.risk.daily_loss_halt,
        max_drawdown_halt=cfg.risk.max_drawdown_halt,
        day_start_equity=float(state.get("day_start_equity") or total_equity),
        peak_equity=float(state.get("peak_equity") or total_equity),
        halted=bool(state.get("halted")),
        halt_reason=str(state.get("halt_reason") or ""),
    )

    # per-coin brokers seeded from persisted state
    brokers: dict[str, PaperBroker] = {}
    strats: dict[str, object] = {}
    coin_states: dict[str, dict] = {}
    for sym in symbols:
        cs = state.setdefault("per_coin", {}).setdefault(sym, _default_coin_state())
        coin_states[sym] = cs
        brokers[sym] = PaperBroker(
            cost=cost,
            equity=sub_capital,  # per-coin capital (will rebalance below)
            pos=int(cs.get("pos") or 0),
            qty=float(cs.get("qty") or 0),
            entry=float(cs.get("entry") or 0),
            symbol=sym,
            order_seq=int(cs.get("order_seq") or 0),
        )
        strats[sym] = make_strategy(cfg)

    all_orders: list[dict] = []
    all_logs: list[dict] = []
    coin_returns: dict[str, float] = {}
    new_last_bar = None

    for sym in symbols:
        try:
            klines = fetch_public_spot_klines(
                symbol=sym, interval=cfg.interval, limit=bars_needed, closed_only=True,
            )
        except Exception as e:
            log.warning("  %s klines fetch failed: %s", sym, e)
            continue
        if len(klines) <= max(cfg.n, 50):
            log.warning("  %s not enough bars: %d", sym, len(klines))
            continue

        bar_id = str(klines.iloc[-1]["open_time"])
        if new_last_bar is None:
            new_last_bar = bar_id
        if sym == symbols[0] and prev_last_bar == bar_id:
            log.info("already processed bar %s", bar_id)
            return {"skipped": True, "bar": bar_id}

        strat = strats[sym]
        closes = klines["close"].to_numpy(dtype=float)
        for i in range(len(closes)):
            strat.on_bar_close(closes[: i + 1])

        sig = strat.curr_sig
        side = 1 if sig > 0 else -1 if sig < 0 else 0
        broker = brokers[sym]
        mark_px = float(closes[-1])

        # risk guard at portfolio level: if halted, force flat
        allowed = risk.check(total_equity)
        target = side if allowed else 0
        if not allowed:
            log.warning("  %s 风控停机: %s", sym, risk.halt_reason)

        if target == 0 and broker.pos == 0 or target == broker.pos:
            orders = []
        else:
            orders = broker.set_position(
                target, mark_px, risk, bar=bar_id, portfolio_equity=total_equity,
            )

        all_orders.extend(o.to_dict() for o in orders)
        coin_returns[sym] = broker.equity / sub_capital - 1.0

        cs = coin_states[sym]
        cs.clear()
        cs.update(
            pos=broker.pos, qty=broker.qty, entry=broker.entry,
            hold_left=getattr(strat, "hold_left", 0),
            curr_sig=float(sig),
            bars_in_trade=getattr(strat, "bars_in_trade", 0),
            last_fit=getattr(strat, "last_fit", None),
            pred_y=getattr(strat, "pred_y", None),
            extreme=getattr(strat, "extreme", 0.0),
            order_seq=broker.order_seq,
        )
        all_logs.append({
            "bar": bar_id, "symbol": sym, "close": mark_px,
            "signal": float(sig), "pos": broker.pos,
            "equity": broker.equity, "halted": risk.halted,
            "n_orders": len(orders), "source": "spot_public",
        })

    if new_last_bar is None:
        return {"error": "no klines for any symbol"}

    # total equity = sum of per-coin equity (each starts at sub_capital)
    total_equity = sum(b.equity for b in brokers.values())
    day = new_last_bar[:10]
    if state.get("day") != day:
        risk.reset_day(total_equity)
        state["day"] = day

    state.clear()
    state.update(
        equity=total_equity,
        per_coin=coin_states,
        peak_equity=risk.peak_equity,
        day=day,
        day_start_equity=risk.day_start_equity,
        halted=risk.halted,
        halt_reason=risk.halt_reason,
        last_bar=new_last_bar,
        n_steps=prev_n_steps + 1,
        n_trades=prev_n_trades + len(all_orders),
        n_orders=prev_n_orders + len(all_orders),
        order_seq=order_seq,
        source="spot_public_3coin",
        note="simulated FILLED orders only; no exchange API; 3-coin breakout portfolio",
    )
    save_state(STATE_PATH, state)
    append_csv(LOG_PATH, all_logs)
    append_csv(ORDERS_PATH, all_orders)

    log.info(
        "bar=%s equity=%.2f orders=%d per_coin=%s halted=%s",
        new_last_bar, total_equity, len(all_orders),
        {s: f"{r*100:+.1f}%" for s, r in coin_returns.items()},
        risk.halted,
    )
    return {
        "bar": new_last_bar, "equity": total_equity,
        "orders": len(all_orders), "halted": risk.halted,
        "per_coin": coin_returns,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loop", type=float, default=0.0,
                   help="poll interval seconds; 0 = single run (cron mode)")
    args = p.parse_args()

    cfg = PassedConfig.load(ROOT / "configs" / "passed.yaml")

    if args.loop <= 0:
        step_once(cfg)
        return

    log.info("live paper loop: polling every %.0fs", args.loop)
    while True:
        try:
            step_once(cfg)
        except Exception as e:
            log.error("step error: %s", e)
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
