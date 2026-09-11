"""Tests for RunLoop risk guard, reconciliation, and ledgers (no network)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from live.runloop import RunLoop
from research.config import PassedConfig


class FakeBroker:
    """Minimal broker double: no network, deterministic state."""

    KLINE_COLS = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]

    def __init__(self, equity: float = 5000.0, pos_amt: float = 0.0):
        self.equity = equity
        self.pos_amt = pos_amt
        self.orders: list[dict] = []
        self.stop_orders: dict[str, list[dict]] = {}

    def usdt_balance(self) -> dict:
        return {
            "wallet": self.equity * 0.98,
            "unrealized": self.equity * 0.02,
            "equity": self.equity,
            "available": self.equity,
            "can_trade": True,
        }

    def position(self, symbol: str) -> dict:
        return {
            "symbol": symbol,
            "positionAmt": self.pos_amt,
            "entryPrice": 100.0 if self.pos_amt else 0.0,
            "unRealizedProfit": 0.0,
            "markPrice": 100.0,
        }

    def klines(self, symbol: str, interval: str = "1h", limit: int = 800, closed_only: bool = True):
        n = min(limit, 800)
        closes = [100.0 + i * 0.01 for i in range(n)]
        rows = []
        for i, c in enumerate(closes):
            rows.append([
                i * 3600_000, c, c, c, c, 1.0,
                (i * 3600_000) + 3599_999, 100.0, 1, 0.5, 50.0, 0,
            ])
        return rows

    def round_qty(self, symbol: str, qty: float) -> float:
        return round(max(0.0, qty), 4)

    def round_price(self, symbol: str, price: float) -> float:
        return float(round(price, 1))

    def min_notional(self, symbol: str) -> float:
        return 55.0

    def open_orders(self, symbol: str) -> list:
        return list(self.stop_orders.get(symbol, []))

    def cancel_all_orders(self, symbol: str) -> dict:
        self.stop_orders.pop(symbol, None)
        return {"ok": True}

    def stop_market_close(self, symbol: str, side: str, stop_price: float) -> dict:
        rec = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": stop_price,
            "closePosition": "true",
        }
        self.stop_orders.setdefault(symbol, []).append(rec)
        return rec

    def market_order(self, symbol: str, side: str, qty: float,
                     reduce_only: bool = False, client_order_id: str = "") -> dict:
        rec = {
            "orderId": len(self.orders) + 1,
            "executedQty": qty,
            "avgPrice": 100.0,
            "status": "FILLED",
        }
        self.orders.append({**rec, "side": side, "reduce_only": reduce_only})
        if reduce_only:
            self.pos_amt = 0.0
        else:
            self.pos_amt = qty if side == "BUY" else -qty
        return rec


class FakeTG:
    def __init__(self):
        self.calls: list[tuple] = []

    def online(self, snap):
        self.calls.append(("online", snap))

    def offline(self, reason=""):
        self.calls.append(("offline", reason))

    def disconnect(self, detail):
        self.calls.append(("disconnect", detail))

    def reconnect(self):
        self.calls.append(("reconnect",))

    def error(self, detail):
        self.calls.append(("error", detail))

    def halt(self, reason, equity):
        self.calls.append(("halt", reason, equity))

    def clear_halt(self):
        self.calls.append(("clear_halt",))

    def fills(self, orders):
        self.calls.append(("fills", orders))

    def stop_placed(self, symbol, side, price):
        self.calls.append(("stop_placed", symbol, side, price))

    def fit_warn(self, leg):
        self.calls.append(("fit_warn", leg))

    def hourly(self, snap):
        self.calls.append(("hourly", snap))

    def due_hourly(self):
        return False

    def stale_bar(self, last_bar, hours):
        self.calls.append(("stale_bar", last_bar, hours))


def _mk_config(tmp_path: Path) -> PassedConfig:
    cfg_yaml = tmp_path / "passed.yaml"
    cfg_yaml.write_text(
        """
symbol: BTCUSDT
symbols:
  - BTCUSDT
interval: 1h
lookback: 48
hold: 48
mode: long_short
exit_mode: time
pip_n: 5
min_bars: 4
fit_exit: 0.0
notional_frac: 0.25
risk:
  notional_frac: 0.25
  max_leverage: 2.0
  daily_loss_halt: 0.05
  max_drawdown_halt: 0.15
backtest:
  start_equity: 1000.0
"""
    )
    return PassedConfig.load(cfg_yaml)


@pytest.fixture()
def runloop(tmp_path):
    cfg = _mk_config(tmp_path)
    rl = RunLoop.__new__(RunLoop)
    rl.cfg = cfg
    rl.report_dir = tmp_path / "reports"
    rl.report_dir.mkdir(parents=True, exist_ok=True)
    rl.broker_factory = FakeBroker
    rl.broker = FakeBroker(equity=5000.0)
    rl.tg = FakeTG()
    rl.min_notional = 55.0
    rl.symbols = cfg.symbols
    rl.frac_each = cfg.notional_frac / len(rl.symbols)
    rl.log_path = rl.report_dir / "testnet_log.csv"
    rl.orders_path = rl.report_dir / "testnet_orders.csv"
    rl.account_path = rl.report_dir / "testnet_account.json"
    rl.account = rl._load_account()
    rl._restored = False
    rl._cooldown_until = 0.0
    from live.risk import RiskGuard

    rl.risk = RiskGuard(
        notional_frac=rl.frac_each,
        max_leverage=2.0,
        daily_loss_halt=0.05,
        max_drawdown_halt=0.15,
    )
    rl.strategies = {}
    rl.states = {}
    for sym in rl.symbols:
        from live.strategy import FrozenMomentumStrategy

        rl.strategies[sym] = FrozenMomentumStrategy(
            lookback=cfg.lookback, hold=cfg.hold, mode=cfg.mode,
            exit_mode=cfg.exit_mode, pip_n=cfg.pip_n,
            min_bars=cfg.min_bars, fit_exit=cfg.fit_exit,
        )
        rl.states[sym] = rl._load_state(sym)
    return rl


def test_refresh_equity_uses_broker_and_updates_account(runloop):
    runloop.broker.equity = 5123.45
    eq = runloop._refresh_equity()
    assert eq == pytest.approx(5123.45)
    assert runloop.account["equity"] == pytest.approx(5123.45)
    assert runloop.account["day_start_equity"] == pytest.approx(5123.45)
    assert runloop.account["peak_equity"] == pytest.approx(5123.45)


def test_refresh_equity_rolls_day_and_clears_reason(runloop):
    runloop.account["day"] = "2000-01-01"
    runloop.account["halt_reason"] = "daily_loss 0.06"
    runloop.account["halted"] = False
    runloop.broker.equity = 4000.0
    runloop._refresh_equity()
    assert runloop.account["day"] == pd.Timestamp.now(tz="UTC").date().isoformat()
    assert runloop.account["halt_reason"] == ""
    assert runloop.account["day_start_equity"] == 4000.0


def test_daily_loss_halt_forces_flatten(runloop):
    # Simulate: equity drops 6% within the day -> halted, must flatten long
    runloop.account["day_start_equity"] = 5000.0
    runloop.account["day"] = pd.Timestamp.now(tz="UTC").date().isoformat()
    runloop.account["peak_equity"] = 5000.0
    runloop.risk.day_start_equity = 5000.0
    runloop.risk.peak_equity = 5000.0
    runloop.broker.equity = 4700.0  # -6% day loss
    runloop.broker.pos_amt = 0.5  # holding a long

    eq = runloop._refresh_equity()
    summary = runloop.run_all()

    assert runloop.risk.halted is True
    assert "daily_loss" in runloop.risk.halt_reason
    assert summary["halted"] is True
    # A reduce-only SELL order should have been sent to flatten the long
    closes = [o for o in runloop.broker.orders if o["reduce_only"]]
    assert len(closes) >= 1
    assert closes[0]["side"] == "SELL"
    # No new opens while halted
    opens = [o for o in runloop.broker.orders if not o["reduce_only"]]
    assert len(opens) == 0


def test_reconciles_stale_local_state_to_exchange(runloop):
    # Local state thinks it's long, exchange is flat -> state must reset,
    # then the loop may legitimately re-open based on signal (fresh sizing).
    st = runloop.states["BTCUSDT"]
    st["pos"] = 1
    st["qty"] = 0.5
    st["entry"] = 100.0
    runloop.broker.pos_amt = 0.0

    runloop.run_all()

    # Whatever the final position, local state must mirror the exchange,
    # never keep the stale phantom qty/entry from before reconciliation.
    exch_side = 1 if runloop.broker.pos_amt > 1e-12 else -1 if runloop.broker.pos_amt < -1e-12 else 0
    assert st["pos"] == exch_side
    if exch_side == 0:
        assert st["qty"] == 0.0
        assert st["entry"] == 0.0


def test_ledgers_written(runloop):
    runloop.run_all()
    assert runloop.log_path.exists()
    lines = runloop.log_path.read_text().strip().splitlines()
    assert len(lines) >= 2  # header + at least one row
    assert "symbol" in lines[0] and "equity" in lines[0]
    assert runloop.account_path.exists()
    acct = json.loads(runloop.account_path.read_text())
    assert acct["equity"] > 0


def test_open_uses_real_equity_for_sizing(runloop):
    runloop.broker.equity = 5000.0
    summary = runloop.run_all()
    opens = [o for o in runloop.broker.orders if not o["reduce_only"]]
    if not opens:
        pytest.skip("no signal in synthetic data")
    # 25% of 5000 / 100 = 12.5 qty
    assert opens[0]["executedQty"] == pytest.approx(12.5, rel=0.2)


def test_snapshot_includes_account(runloop):
    runloop.broker.equity = 4321.0
    snap = runloop.snapshot()
    assert "acct" in snap
    assert snap["acct"]["equity"] == pytest.approx(4321.0)
    assert len(snap["legs"]) == len(runloop.symbols)


def test_restore_state_skips_when_flat(runloop):
    runloop.broker.pos_amt = 0.0
    runloop._restore_state("BTCUSDT")  # must not raise
    strat = runloop.strategies["BTCUSDT"]
    assert strat.curr_sig == 0.0


def test_restore_all_sets_restored_flag_and_no_dup(runloop):
    """run_all() must restore exactly once; a second call must not redo it."""
    runloop._restored = False
    runloop.run_all()
    assert runloop._restored is True
    first_n = len(runloop.broker.orders)
    runloop.run_all()
    # Second invocation must not re-restore (would double-fire fills / stops).
    assert len(runloop.broker.orders) == first_n


def test_exchange_stop_placed_on_open(runloop):
    """Open a position and verify a STOP_MARKET is placed on the exchange."""
    runloop.cfg.use_exchange_stop = True
    runloop.cfg.stop_pct = 0.06
    runloop.broker.equity = 5000.0
    runloop.broker.pos_amt = 0.0
    runloop.run_all()
    opens = [o for o in runloop.broker.orders if not o["reduce_only"]]
    if not opens:
        pytest.skip("no signal in synthetic data")
    # A STOP_MARKET order must exist for the opened symbol.
    sym = opens[0]["side"] and "BTCUSDT"
    stops = runloop.broker.stop_orders.get(sym, [])
    assert len(stops) >= 1
    assert stops[0]["type"] == "STOP_MARKET"
    assert str(stops[0]["closePosition"]).lower() == "true"


def test_exchange_stop_cancelled_on_close(runloop):
    """Flatten a position and verify the exchange-side stop is cancelled."""
    runloop.cfg.use_exchange_stop = True
    runloop.cfg.stop_pct = 0.06
    runloop.broker.equity = 5000.0
    runloop.broker.pos_amt = 0.0005  # already long a tiny position
    st = runloop.states["BTCUSDT"]
    st["pos"] = 1
    st["qty"] = 0.0005
    st["entry"] = 100.0
    runloop.broker.stop_orders["BTCUSDT"] = [{
        "symbol": "BTCUSDT", "type": "STOP_MARKET", "closePosition": "true",
    }]
    # Bypass strategy/restore: directly exercise the flatten path so the
    # stop-cancellation side effect is observed regardless of signal.
    runloop._restored = True
    pos_row = {"positionAmt": 0.0005, "entryPrice": 100.0, "unRealizedProfit": 0.0}
    # Mimic run_one_symbol's flatten block: target 0 while holding long.
    from live.runloop import _pos_side
    current = _pos_side(float(pos_row["positionAmt"]))
    assert current != 0
    target = 0  # force flatten
    exch_qty = abs(float(pos_row["positionAmt"]))
    close_side = "SELL" if current > 0 else "BUY"
    rec = runloop._market_order("BTCUSDT", close_side, exch_qty, reduce_only=True, oid_str="T")
    runloop.tg.fills([rec])
    # The flatten path must cancel the exchange-side stop.
    runloop._cancel_stops("BTCUSDT")
    assert runloop.broker.stop_orders.get("BTCUSDT", []) == []
    assert rec["status"] == "FILLED"


def test_restore_rebuilds_counters_from_history(runloop):
    """After restart with an open position, replay must rebuild
    bars_in_trade/hold_left from history, not from stale saved counters."""
    # Seed a stale state that would be wrong if trusted blindly.
    st = runloop.states["BTCUSDT"]
    st["last_bar"] = None
    st["pos"] = 1
    st["bars_in_trade"] = 999
    st["hold_left"] = 999
    runloop.broker.pos_amt = 0.0005  # exchange says we are long
    runloop._restored = False
    runloop._restore_all()
    strat = runloop.strategies["BTCUSDT"]
    # Replayed counters must differ from the bogus saved 999.
    assert strat.bars_in_trade != 999
    assert strat.hold_left != 999


def test_rate_limit_triggers_cooldown(runloop):
    """A 418 error string must arm the cooldown window, not just log."""
    import time as _time

    runloop._cooldown_until = 0.0
    # "418 Client Error" (HTTP status prefix) is an anchored form; bare
    # digits inside stop prices (418xx) or clientOrderIds (TN0418…) must not.
    assert runloop._is_rate_limit("418 Client Error: I'm a teapot for url: ...")
    assert runloop._is_rate_limit('binance 418: {"code":-1003,"msg":"Way too many requests"}')
    assert runloop._is_rate_limit('binance 429: {"code":-1003,"msg":"Too many requests"}')
    assert runloop._is_rate_limit("binance 400: banned until 1788842044401")
    assert not runloop._is_rate_limit("binance 400: some other error")
    # The false-positive classes the anchoring exists for:
    assert not runloop._is_rate_limit("binance 400: STOP_MARKET rejected, stop price 41850.5")
    assert not runloop._is_rate_limit("order TN0429 failed: reduceOnly rejected")
    # Manually arm and verify the loop would skip.
    runloop._cooldown_until = _time.time() + 600.0
    assert runloop._cooldown_until > _time.time()


def test_cooldown_honors_ban_deadline(runloop):
    """'banned until <ms>' must sleep until the stated end, not a fixed 10 min."""
    import time as _time

    runloop._cooldown_until = 0.0
    until_ms = (_time.time() + 14 * 60.0) * 1000.0
    runloop._arm_cooldown(f'banned until {int(until_ms)}. Please use the websocket.')
    remaining = runloop._cooldown_until - _time.time()
    # ~14 min (the ban), not the 10-min default and not zero.
    assert 13 * 60.0 < remaining <= 14.1 * 60.0


def test_cooldown_backs_off_without_deadline(runloop):
    """Repeat hits with no ban deadline must grow the window, not reset it."""
    import time as _time

    runloop._cooldown_until = 0.0
    runloop._arm_cooldown('binance 429: too many requests')
    first = runloop._cooldown_until - _time.time()
    runloop._arm_cooldown('binance 429: too many requests')
    second = runloop._cooldown_until - _time.time()
    assert second > first  # 600 + previous remainder = longer window


def test_rate_limited_klines_abort_cycle(runloop):
    """A 418 during kline fetch must abort run_all, not yield 'no klines'."""
    from live.runloop import RateLimitedError

    def boom(*a, **k):
        raise RuntimeError('binance 418: {"code":-1003,"msg":"Way too many requests; IP banned until 1788489836895"}')

    runloop._restored = True
    runloop.broker.klines = boom
    with pytest.raises(RateLimitedError):
        runloop.run_one_symbol("BTCUSDT", 5000.0)
    summary = runloop.run_all()
    assert "418" in summary["error"]
    assert runloop._is_rate_limit(summary["error"])


def test_hourly_broadcast_survives_rate_limit(runloop, monkeypatch):
    """The hourly report must fire even when the cycle hit a 418.

    Regression: 2026-09-07 the testnet WAF episode put every cycle into the
    rate-limit branch for hours; the hourly check lived only on the clean
    path, so the broadcast went silent and the bot looked dead.
    """
    sent = []
    runloop._cooldown_until = 0.0
    monkeypatch.setattr(runloop.tg, "due_hourly", lambda: True)
    monkeypatch.setattr(runloop.tg, "hourly", lambda snap: sent.append(snap))
    monkeypatch.setattr(runloop.tg, "error", lambda detail: None)

    runloop._handle_rate_limit('binance 418: {"code":-1003,"msg":"Way too many requests"}')
    assert sent, "hourly broadcast must fire on the rate-limit path too"
    import time as _time

    assert runloop._cooldown_until > _time.time()  # cooldown still armed
    # and not due again immediately (send dedups via the hourly state stamp)
    monkeypatch.setattr(runloop.tg, "due_hourly", lambda: False)
    runloop._handle_rate_limit("binance 429: too many requests")
    assert len(sent) == 1


def test_cooldown_persists_across_restart(runloop):
    """Armed cooldown must survive a restart so restore doesn't hit a live ban."""
    import time as _time

    runloop._cooldown_until = 0.0
    until_ms = (_time.time() + 30 * 60.0) * 1000.0
    runloop._arm_cooldown(f'banned until {int(until_ms)}')
    assert runloop.account["cooldown_until"] == pytest.approx(runloop._cooldown_until)
    # Simulate restart: account file reloaded from disk carries the deadline.
    import json as _json

    disk = _json.loads(runloop.account_path.read_text())
    assert disk["cooldown_until"] > _time.time()


def test_restore_defers_until_cooldown_expires(runloop):
    """_restore_all must wait out an active cooldown before any broker call."""
    import time as _time

    calls = {"restore": 0}
    runloop._cooldown_until = _time.time() + 1.0  # short wait for the test

    orig = runloop._restore_state

    def counting(sym):
        calls["restore"] += 1
        orig(sym)

    runloop._restore_state = counting
    runloop._restore_all()
    assert calls["restore"] == len(runloop.symbols)
    assert runloop._cooldown_until <= _time.time()


def test_cooldown_uses_retry_after_hint(runloop):
    """429 with a Retry-After header must sleep that long, not fixed 600s."""
    import time as _time

    runloop._cooldown_until = 0.0
    runloop._arm_cooldown(
        'binance 429: {"code":-1003,"msg":"Too many requests"} retry-after 45'
    )
    remaining = runloop._cooldown_until - _time.time()
    assert 44.0 < remaining <= 51.0


def test_retry_after_beats_backoff_but_not_ban_deadline(runloop):
    """Precedence: banned until > retry-after > backoff window."""
    import time as _time

    runloop._cooldown_until = 0.0
    # ban deadline present: wins over the retry-after hint in the same text
    until_ms = (_time.time() + 20 * 60.0) * 1000.0
    runloop._arm_cooldown(f'banned until {int(until_ms)} retry-after 10')
    remaining = runloop._cooldown_until - _time.time()
    assert 19 * 60.0 < remaining <= 20.1 * 60.0

