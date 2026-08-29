"""Main loop for live trading, shared across testnet and future live modes.

Encapsulates the signal-processing loop, state management, risk guard,
account bookkeeping, Telegram alerts, and clean shutdown handling.
"""
from __future__ import annotations

import contextlib
import json
import signal
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from live.notify import TelegramNotifier
from live.risk import RiskGuard
from live.strategy import FrozenMomentumStrategy
from research.config import PassedConfig
from research.log import get_logger

log = get_logger("runloop")

LOG_COLUMNS = [
    "bar", "symbol", "mark", "signal", "pos", "qty", "equity", "n_orders", "halted",
]
ORDER_COLUMNS = [
    "client_order_id", "exchange_id", "bar", "symbol", "side", "action",
    "reduce_only", "qty", "avg_price", "status", "ts", "pnl", "pnl_pct",
]


def _first_unprocessed_idx(klines: pd.DataFrame, last_bar: str | None) -> int:
    """Index of the first closed bar not yet applied to strategy state."""
    n = len(klines)
    if n == 0:
        return 0
    if not last_bar:
        return n - 1
    try:
        target = pd.Timestamp(last_bar)
    except Exception:
        return n - 1
    for i in range(n - 1, -1, -1):
        bt = klines.iloc[i].get("open_time")
        if bt is not None and pd.Timestamp(bt) <= target:
            return i + 1
    return 0


def _latest_last_bar(states: dict[str, dict]) -> str | None:
    """Return the most recent last_bar across all symbol states."""
    best, best_ts = None, None
    for _sym, st in states.items():
        lb = st.get("last_bar")
        if lb:
            try:
                ts = pd.Timestamp(lb)
            except Exception:
                continue
            if best_ts is None or ts > best_ts:
                best, best_ts = lb, ts
    return best


def _bar_lag_hours(last_bar: str | None) -> float:
    """Return hours since the last bar timestamp."""
    if not last_bar:
        return 0.0
    try:
        bt = pd.Timestamp(last_bar)
        return (pd.Timestamp.now(tz="UTC") - bt).total_seconds() / 3600.0
    except Exception:
        return 0.0


def _pos_side(amt: float) -> int:
    """Return +1/0/-1 from a position amount, with epsilon tolerance."""
    if amt > 1e-12:
        return 1
    if amt < -1e-12:
        return -1
    return 0


def _oid(seq: int, kind: str, symbol: str) -> str:
    """Generate a short client order ID for testnet."""
    tag = symbol.replace("USDT", "")[:3]
    return f"TN{seq:04d}{tag}{kind[:3]}"[:36]


def _append_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Append rows to a CSV ledger file, writing headers on first write."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


class RunLoop:
    """Manages the per-symbol strategy state, broker interaction, and alerts.

    Parameters
    ----------
    config : PassedConfig
        Frozen strategy configuration.
    report_dir : Path
        Directory for state files and logs.
    broker_factory : Callable
        Callable that returns a broker instance (e.g., BinanceUSDMBroker).
    tg : TelegramNotifier
        Notification dispatcher.
    min_notional : float
        Minimum USD notional for opening a position (skip if below).
    """

    def __init__(
        self,
        config: PassedConfig,
        report_dir: Path,
        broker_factory: Callable[[], Any],
        tg: TelegramNotifier,
        min_notional: float = 55.0,
    ):
        self.cfg = config
        self.report_dir = report_dir
        self.broker_factory = broker_factory
        self.broker = broker_factory()
        self.tg = tg
        self.min_notional = min_notional
        self.symbols = config.symbols
        self.frac_each = (
            config.notional_frac / len(self.symbols) if config.symbols else config.notional_frac
        )

        self.log_path = report_dir / "testnet_log.csv"
        self.orders_path = report_dir / "testnet_orders.csv"
        self.account_path = report_dir / "testnet_account.json"

        self.account: dict = self._load_account()
        self.risk = RiskGuard(
            notional_frac=self.frac_each,
            max_leverage=float(config.risk.max_leverage),
            daily_loss_halt=float(config.risk.daily_loss_halt),
            max_drawdown_halt=float(config.risk.max_drawdown_halt),
            day_start_equity=float(self.account.get("day_start_equity") or 0),
            peak_equity=float(self.account.get("peak_equity") or 0),
            halted=bool(self.account.get("halted")),
            halt_reason=str(self.account.get("halt_reason") or ""),
        )

        self.strategies: dict[str, FrozenMomentumStrategy] = {}
        self.states: dict[str, dict] = {}
        for sym in self.symbols:
            self.strategies[sym] = FrozenMomentumStrategy(
                lookback=config.lookback,
                hold=config.hold,
                mode=config.mode,
                exit_mode=config.exit_mode,
                pip_n=config.pip_n,
                min_bars=config.min_bars,
                fit_exit=config.fit_exit,
                stop_pct=config.stop_pct,
                trail_arm=config.trail_arm,
                trail_giveback=config.trail_giveback,
            )
            self.states[sym] = self._load_state(sym)

    def _state_path(self, symbol: str) -> Path:
        """Path to the per-symbol state file."""
        d = self.report_dir / "testnet"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol}_state.json"

    def _legacy_state_path(self) -> Path:
        """Path to the legacy single-file state (BTCUSDT only)."""
        return self.report_dir / "testnet_state.json"

    def _load_account(self) -> dict:
        """Load persisted account bookkeeping, or defaults."""
        if self.account_path.exists():
            return json.loads(self.account_path.read_text())
        return {
            "peak_equity": 0.0,
            "day": None,
            "day_start_equity": 0.0,
            "halted": False,
            "halt_reason": "",
            "order_seq": 0,
            "equity": 0.0,
            "wallet": 0.0,
            "unrealized": 0.0,
        }

    def _save_account(self) -> None:
        """Persist account bookkeeping to disk."""
        self.account_path.parent.mkdir(parents=True, exist_ok=True)
        self.account_path.write_text(json.dumps(self.account, indent=2, default=float))

    def _refresh_equity(self) -> float:
        """Fetch equity from the broker, roll the day, and update peak."""
        bal = self.broker.usdt_balance()
        equity = float(bal.get("equity") if bal.get("equity") is not None else bal.get("wallet") or 0.0)
        day = pd.Timestamp.now(tz="UTC").date().isoformat()
        if self.account.get("day") != day:
            self.account["day"] = day
            self.account["day_start_equity"] = equity
            if not self.account.get("halted"):
                self.account["halt_reason"] = ""
            self.risk.day_start_equity = equity
        if equity > float(self.account.get("peak_equity") or 0):
            self.account["peak_equity"] = equity
            self.risk.peak_equity = equity
        self.account["equity"] = equity
        self.account["wallet"] = float(bal.get("wallet") or equity)
        self.account["unrealized"] = float(bal.get("unrealized") or 0)
        return equity

    def _load_state(self, symbol: str) -> dict:
        """Load persisted state for a symbol, or return defaults."""
        p = self._state_path(symbol)
        if p.exists():
            return json.loads(p.read_text())
        if symbol == "BTCUSDT":
            leg = self._legacy_state_path()
            if leg.exists():
                st = json.loads(leg.read_text())
                p.write_text(json.dumps(st, indent=2, default=float))
                return st
        return {
            "symbol": symbol,
            "hold_left": 0,
            "curr_sig": 0.0,
            "last_bar": None,
            "n_steps": 0,
            "n_orders": 0,
            "order_seq": 0,
            "pred_pips": None,
            "pos": 0,
            "qty": 0.0,
            "entry": 0.0,
            "bars_in_trade": 0,
            "extreme": 0.0,
        }

    def _save_state(self, symbol: str, state: dict) -> None:
        """Persist state for a symbol to disk."""
        p = self._state_path(symbol)
        p.write_text(json.dumps(state, indent=2, default=float))
        if symbol == "BTCUSDT":
            self._legacy_state_path().write_text(json.dumps(state, indent=2, default=float))

    def _restore_state(self, symbol: str) -> None:
        """Restore intraday strategy state from a previous run.

        Replays historical bars through the strategy to reconstruct PIP
        state and bar counters.
        """
        st = self.states[symbol]
        strat = self.strategies[symbol]
        broker = self.broker
        try:
            pos_amt = float(broker.position(symbol).get("positionAmt") or 0)
        except Exception:
            pos_amt = 0.0
        pos = _pos_side(pos_amt)
        if pos == 0:
            log.info("  %s 无持仓，跳过恢复", symbol)
            return
        bar = st.get("last_bar")
        if bar:
            try:
                klines = self._fetch_klines(symbol, limit=100)
                idx = _first_unprocessed_idx(klines, bar)
                closes = klines["close"].to_numpy(dtype=float)
                strat.curr_sig = float(st.get("curr_sig") or 0)
                strat.hold_left = int(st.get("hold_left") or 0)
                strat.bars_in_trade = int(st.get("bars_in_trade") or 0)
                strat.entry_price = float(st.get("entry") or 0)
                strat.extreme = float(st.get("extreme") or strat.entry_price)
                pred = st.get("pred_pips")
                if pred:
                    strat.pred_y = [float(p["y"]) for p in pred]
                strat.replay_from(closes, idx)
                log.info(
                    "  %s 恢复完成: pos=%d bars=%d hold_left=%d",
                    symbol, pos, strat.bars_in_trade, strat.hold_left,
                )
            except Exception as e:
                log.warning("  %s 恢复失败: %s", symbol, e)

    def _fetch_klines(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """Fetch klines from broker, returning a DataFrame with standard columns."""
        try:
            raw = self.broker.klines(symbol, self.cfg.interval, limit=limit)
        except Exception as e:
            log.warning("  %s klines fetch failed: %s", symbol, e)
            return pd.DataFrame()
        klines = pd.DataFrame(raw, columns=self.broker.KLINE_COLS)
        klines["open_time"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)
        klines["close_time"] = pd.to_datetime(klines["close_time"], unit="ms", utc=True)
        return klines

    def _round_qty(self, symbol: str, qty: float) -> float:
        """Round quantity according to broker filters."""
        return self.broker.round_qty(symbol, qty)

    def _market_order(
        self, symbol: str, side: str, qty: float,
        reduce_only: bool = False, oid_str: str = "",
    ) -> dict:
        """Place a market order via the broker."""
        return self.broker.market_order(
            symbol, side, qty, reduce_only=reduce_only, client_order_id=oid_str,
        )

    def run_one_symbol(self, symbol: str, equity: float) -> dict:
        """Process one bar for one symbol. Returns summary dict."""
        st = self.states[symbol]
        strat = self.strategies[symbol]

        klines = self._fetch_klines(symbol, limit=self.cfg.lookback + 10)
        if klines.empty:
            return {"symbol": symbol, "error": "no klines"}

        orders: list[dict] = []
        results: dict = {"symbol": symbol, "orders": orders}

        bar_row = klines.iloc[-1]
        bar_id = str(bar_row["open_time"])
        mark = float(bar_row["close"])

        if st.get("last_bar") == bar_id:
            log.debug("  %s 已处理 %s", symbol, bar_id)
            return results

        inject = st.get("last_bar") is None
        idx = _first_unprocessed_idx(klines, st.get("last_bar"))
        closes = klines["close"].to_numpy(dtype=float)

        # Run strategy
        if inject:
            strat.replay_from(closes, 0)
        else:
            for i in range(idx, len(closes)):
                strat.on_bar_close(closes[: i + 1])
        sig = strat.curr_sig

        # Risk guard: if halted, force target side to 0 (flatten on next step)
        allowed = self.risk.check(equity)
        self.account["halted"] = self.risk.halted
        self.account["halt_reason"] = self.risk.halt_reason
        target = int(np.sign(sig)) if allowed else 0
        if not allowed:
            log.warning("  %s 风控停机: %s", symbol, self.risk.halt_reason)

        # Reconcile with exchange truth (post-crash safety)
        try:
            pos_row = self.broker.position(symbol)
        except Exception as e:
            return {"symbol": symbol, "error": f"position query failed: {e}"}
        exch_amt = float(pos_row.get("positionAmt") or 0)
        current = _pos_side(exch_amt)
        entry_px = float(pos_row.get("entryPrice") or 0)
        upnl = float(pos_row.get("unRealizedProfit") or 0)
        exch_qty = abs(exch_amt)
        if exch_qty == 0.0:
            exch_qty = abs(float(st.get("qty") or 0))
        if current == 0 and st.get("pos") != 0:
            log.warning("  %s 交易所仓位与状态不一致（交易所空仓），重置本地状态", symbol)
            st["pos"] = 0
            st["qty"] = 0.0
            st["entry"] = 0.0
        elif current != 0 and st.get("pos") != current:
            log.warning("  %s 交易所仓位与状态不一致（%s vs %s），以交易所为准", symbol, current, st.get("pos"))
            st["pos"] = current
            st["qty"] = exch_qty
            st["entry"] = entry_px

        now = datetime.now(UTC).isoformat()

        def send(side_txt: str, qty: float, reduce_only: bool, action: str) -> dict:
            nonlocal st
            st["order_seq"] = int(st.get("order_seq") or 0) + 1
            cid = _oid(st["order_seq"], action, symbol)
            raw = self._market_order(symbol, side_txt, qty, reduce_only=reduce_only, oid_str=cid)
            rec = {
                "client_order_id": cid,
                "exchange_id": raw.get("orderId"),
                "bar": bar_id,
                "symbol": symbol,
                "side": side_txt,
                "action": action,
                "reduce_only": reduce_only,
                "qty": float(raw.get("executedQty") or qty),
                "avg_price": float(raw.get("avgPrice") or 0),
                "status": raw.get("status"),
                "ts": now,
            }
            orders.append(rec)
            return rec

        # Flatten if we hold but shouldn't (target==0 or direction flipped)
        if current != 0 and (target != current or not allowed):
            close_side = "SELL" if current > 0 else "BUY"
            notional0 = entry_px * exch_qty
            pnl_pct = (upnl / notional0) if (upnl != 0.0 and notional0 > 0) else None
            rec = send(close_side, exch_qty, True, "CLOSE")
            if pnl_pct is not None:
                rec["pnl"] = float(upnl)
                rec["pnl_pct"] = float(pnl_pct)
            self.tg.fills([rec])
            log.info("  %s 平仓 %s qty=%g upnl=%.2f", symbol, close_side, exch_qty, upnl)
            current = 0
            exch_qty = 0.0
            st["pos"] = 0
            st["qty"] = 0.0
            st["entry"] = 0.0
            st["extreme"] = 0.0

        # Open if we have a target but no position
        if target != 0 and current == 0 and self.risk.check(equity):
            raw_qty = self.risk.target_qty(equity, mark)
            qty = self._round_qty(symbol, raw_qty)
            if qty <= 0 or qty * mark < max(self.min_notional, self.broker.min_notional(symbol)):
                log.info(
                    "  %s notional %.2f < min, skip open", symbol, (qty or 0.0) * mark,
                )
            else:
                open_side = "BUY" if target > 0 else "SELL"
                rec = send(open_side, qty, False, "OPEN")
                self.tg.fills([rec])
                log.info("  %s 开仓 %s qty=%g", symbol, open_side, qty)
                # Record frozen prediction shape from the bar window
                look = min(24, len(closes))
                if look >= self.cfg.pip_n:
                    from research.pips import find_pips

                    _, py = find_pips(closes[-look:], self.cfg.pip_n, 3)
                    st["pred_pips"] = [{"i": int(i), "y": float(y)} for i, y in zip(range(look), py, strict=False)]
                st["pos"] = target
                st["qty"] = qty
                st["entry"] = mark
                st["extreme"] = mark
                st["hold_left"] = strat.hold_left
                st["bars_in_trade"] = strat.bars_in_trade

        # Update state
        st["last_bar"] = bar_id
        st["curr_sig"] = float(sig)
        st["hold_left"] = strat.hold_left
        st["bars_in_trade"] = strat.bars_in_trade
        st["n_steps"] = st.get("n_steps", 0) + 1
        st["last_fit"] = strat.last_fit
        st["mark"] = mark
        st["equity"] = equity
        if st.get("pos"):
            st["extreme"] = st.get("extreme") or st.get("entry") or 0.0
        self._save_state(symbol, st)

        _append_csv(
            self.log_path,
            [{
                "bar": bar_id,
                "symbol": symbol,
                "mark": mark,
                "signal": float(sig),
                "pos": int(st.get("pos") or 0),
                "qty": float(st.get("qty") or 0),
                "equity": equity,
                "n_orders": len(orders),
                "halted": self.risk.halted,
            }],
            LOG_COLUMNS,
        )
        _append_csv(self.orders_path, orders, ORDER_COLUMNS)

        return results

    def run_all(self) -> dict:
        """Process one bar for all symbols. Returns aggregated summary."""
        summary: dict = {"symbols": {}, "orders": []}
        try:
            equity = self._refresh_equity()
        except Exception as e:
            return {"error": f"equity fetch failed: {e}", "symbols": {}, "orders": []}
        for sym in self.symbols:
            try:
                res = self.run_one_symbol(sym, equity)
                summary["symbols"][sym] = res
                summary["orders"].extend(res.get("orders", []))
            except Exception as e:
                log.error("  %s error: %s", sym, e)
                summary["symbols"][sym] = {"symbol": sym, "error": str(e)}
            # Refresh equity after any fills so sizing stays accurate
            with contextlib.suppress(Exception):
                equity = self._refresh_equity()
        self._save_account()
        summary["equity"] = equity
        summary["halted"] = self.risk.halted
        summary["halt_reason"] = self.risk.halt_reason
        return summary

    def snapshot(self) -> dict:
        """Build a snapshot of current state for Telegram alerts."""
        legs = []
        try:
            bal = self.broker.usdt_balance()
            self.account["equity"] = float(
                bal.get("equity") if bal.get("equity") is not None else bal.get("wallet") or 0
            )
            self.account["wallet"] = float(bal.get("wallet") or 0)
            self.account["unrealized"] = float(bal.get("unrealized") or 0)
            if self.account["equity"] > float(self.account.get("peak_equity") or 0):
                self.account["peak_equity"] = self.account["equity"]
                self.risk.peak_equity = self.account["equity"]
        except Exception as e:
            log.warning("snapshot equity fetch failed: %s", e)
        for sym in self.symbols:
            st = self.states[sym]
            strat = self.strategies[sym]
            legs.append({
                "symbol": sym,
                "pos": int(st.get("pos") or 0),
                "qty": float(st.get("qty") or 0),
                "entry": float(st.get("entry") or 0),
                "mark": float(st.get("mark") or 0),
                "fit": strat.last_fit,
                "bars_in_trade": strat.bars_in_trade,
                "hold_left": strat.hold_left,
            })
        return {"acct": self.account, "legs": legs}

    def run_loop(self, loop_minutes: float) -> None:
        """Run the main loop, processing bars on a timer."""
        sleep_s = max(15.0, loop_minutes * 60.0)
        log.info("loop mode: every %.1f min, sleep=%gs", loop_minutes, sleep_s)

        stopping = {"done": False}

        def _stop(signum, _frame):
            if stopping["done"]:
                return
            stopping["done"] = True
            with contextlib.suppress(Exception):
                self.tg.offline(f"signal {signum}")
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)

        # Restore in-flight strategy state (PIP shape, entry, extreme) so a
        # restart doesn't lose exit context for open positions.
        for sym in self.symbols:
            with contextlib.suppress(Exception):
                self._restore_state(sym)

        with contextlib.suppress(Exception):
            self.tg.online(self.snapshot())

        try:
            while True:
                try:
                    summary = self.run_all()
                    err = str(summary.get("error") or "")
                    if err:
                        net = any(
                            x in err.lower()
                            for x in ("unreachable", "timeout", "timed out", "connection")
                        )
                        if net:
                            self.tg.disconnect(err)
                        else:
                            self.tg.reconnect()
                            self.tg.error(err)
                    else:
                        self.tg.reconnect()
                        self._handle_alerts(summary)
                        if self.tg.due_hourly():
                            self.tg.hourly(self.snapshot())
                except Exception as e:
                    log.error("step error: %s", e)
                    msg = str(e)
                    net = any(
                        x in msg.lower()
                        for x in ("unreachable", "timeout", "timed out", "connection")
                    )
                    if net:
                        self.tg.disconnect(msg)
                    else:
                        self.tg.error(msg)
                time.sleep(sleep_s)
        except KeyboardInterrupt:
            _stop(signal.SIGINT, None)

    def _handle_alerts(self, summary: dict) -> None:
        """Dispatch alerts for fills, warnings, and stale bars."""
        if self.risk.halted:
            self.tg.halt(self.risk.halt_reason, float(self.account.get("equity") or 0))
        else:
            self.tg.clear_halt()
        for sym, _res in summary.get("symbols", {}).items():
            st = self.states[sym]
            self.tg.fit_warn({
                "symbol": sym,
                "pos": int(st.get("pos") or 0),
                "qty": float(st.get("qty") or 0),
                "entry": float(st.get("entry") or 0),
                "mark": float(st.get("mark") or 0),
                "fit": self.strategies[sym].last_fit,
                "bars_in_trade": self.strategies[sym].bars_in_trade,
                "hold_left": self.strategies[sym].hold_left,
            })
        last_bar = _latest_last_bar(self.states)
        self.tg.stale_bar(last_bar, _bar_lag_hours(last_bar))
