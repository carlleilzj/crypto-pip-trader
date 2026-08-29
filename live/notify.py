"""Telegram alerts. Token/chat from env. Never blocks trading on send failure."""
from __future__ import annotations

import json
import os
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from live.broker_binance import load_env_file

TG_API = "https://api.telegram.org/bot{token}/sendMessage"
STATE_NAME = "notify_state.json"
MAX_LEN = 3900
FIT_WARN = 0.20
FIT_OK = 0.50


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(ts: datetime | None = None) -> str:
    return (ts or _now()).strftime("%Y-%m-%d %H:%M:%S 世界时")


def _fmt_px(x: float) -> str:
    ax = abs(float(x))
    if ax >= 1000:
        return f"{x:,.1f}"
    if ax >= 10:
        return f"{x:,.2f}"
    return f"{x:,.4f}"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _side_txt(pos: int) -> str:
    if pos > 0:
        return "多"
    if pos < 0:
        return "空"
    return "空仓"


def _zh_action(action: str) -> str:
    return {"OPEN": "开仓", "CLOSE": "平仓"}.get(str(action or "").upper(), str(action or ""))


def _zh_side(side: str) -> str:
    return {"BUY": "买入", "SELL": "卖出"}.get(str(side or "").upper(), str(side or ""))


def _zh_status(status: str) -> str:
    return {
        "FILLED": "已成交",
        "NEW": "已提交",
        "PARTIALLY_FILLED": "部分成交",
        "CANCELED": "已撤销",
        "EXPIRED": "已过期",
        "REJECTED": "已拒绝",
    }.get(str(status or "").upper(), str(status or "未知"))


def _zh_halt(reason: str) -> str:
    r = (reason or "").strip()
    if r.startswith("max_dd"):
        return "最大回撤超限 " + r.split(None, 1)[-1]
    if r.startswith("daily_loss"):
        return "日亏损超限 " + r.split(None, 1)[-1]
    if r in ("halt", ""):
        return "风控触发"
    return r


def _zh_exit(reason: str) -> str:
    r = str(reason or "").strip()
    low = r.lower()
    if r in ("15", "signal 15") or "sigterm" in low or low.endswith(" 15"):
        return "服务停止"
    if r in ("2", "signal 2") or "sigint" in low or low.endswith(" 2"):
        return "手动中断"
    if "systemd" in low and "stop" in low:
        return "服务停止"
    return r


def fit_label(fit: float | None, bars_in_trade: int, min_bars: int = 24) -> str:
    if fit is None:
        if bars_in_trade < min_bars:
            return f"形态冷却 {bars_in_trade}/{min_bars}小时"
        return "形态拟合 —"
    pct = fit * 100.0
    if fit < 0:
        tag = "走样，应出场"
    elif fit < FIT_WARN:
        tag = "接近走样"
    elif fit < 0.50:
        tag = "形态转弱"
    else:
        tag = "形态完好"
    return f"拟合 {pct:.0f}% {tag}"


def format_leg(leg: dict, min_bars: int = 24) -> str:
    sym = str(leg.get("symbol") or "")
    pos = int(leg.get("pos") or 0)
    if pos == 0:
        return f"  {sym} 空仓"
    qty = float(leg.get("qty") or 0)
    entry = float(leg.get("entry") or 0)
    mark = float(leg.get("mark") or 0)
    upnl = float(leg.get("upnl") or 0)
    if upnl == 0.0 and entry and mark and qty:
        upnl = pos * qty * (mark - entry)
    hold = leg.get("hold_left")
    hold_s = f"  剩余持有 {int(hold)}小时" if hold is not None else ""
    return (
        f"  {sym} {_side_txt(pos)} {qty:g} @ {_fmt_px(entry)}  "
        f"现价 {_fmt_px(mark)}  浮盈 {upnl:+.2f}  "
        f"{fit_label(leg.get('fit'), int(leg.get('bars_in_trade') or 0), min_bars)}"
        f"{hold_s}"
    )


def format_hourly(snap: dict) -> str:
    acct = snap.get("acct") or {}
    legs = snap.get("legs") or []
    eq = float(acct.get("equity") or 0)
    wallet = float(acct.get("wallet") or 0)
    unreal = float(acct.get("unrealized") or 0)
    peak = float(acct.get("peak_equity") or 0)
    day0 = float(acct.get("day_start_equity") or 0)
    dd = (1.0 - eq / peak) if peak > 0 else 0.0
    day_pnl = (eq / day0 - 1.0) if day0 > 0 else 0.0
    halted = bool(acct.get("halted"))
    ping_ok = bool(snap.get("ping_ok", True))
    last_bar = snap.get("last_bar") or "—"
    health = "正常" if ping_ok and not halted else ("停机" if halted else "断线/异常")
    lines = [
        f"⏱ 整点汇报 {_iso()}",
        f"健康 {health}  主机 {snap.get('host') or socket.gethostname()}",
        f"上次K线 {last_bar}  每{int(snap.get('loop_min') or 5)}分钟检查一轮",
        f"权益 {eq:.2f}  钱包 {wallet:.2f}  浮盈 {unreal:+.2f}",
        f"日盈亏 {_fmt_pct(day_pnl)}  回撤 {dd * 100:.2f}%  "
        f"停机 {'是，' + _zh_halt(str(acct.get('halt_reason') or '')) if halted else '否'}",
        "持仓:",
    ]
    if not legs:
        lines.append("  （无状态）")
    else:
        lines.extend(format_leg(x) for x in legs)
    return "\n".join(lines)


def format_online(snap: dict) -> str:
    acct = snap.get("acct") or {}
    lines = [
        f"🟢 盯仓上线 {_iso()}",
        f"主机 {snap.get('host') or socket.gethostname()}",
        f"权益 {float(acct.get('equity') or 0):.2f}  浮盈 {float(acct.get('unrealized') or 0):+.2f}",
        "持仓:",
    ]
    legs = snap.get("legs") or []
    lines.extend(format_leg(x) for x in legs) if legs else lines.append("  （读取仓位失败则见下轮）")
    return "\n".join(lines)


def format_offline(reason: str = "") -> str:
    extra = f"\n原因 {_zh_exit(reason)}" if reason else ""
    return f"🔴 盯仓进程退出 {_iso()}{extra}"


def format_disconnect(detail: str) -> str:
    return f"🟠 交易所断线 {_iso()}\n{detail[:500]}"


def format_reconnect() -> str:
    return f"🟢 交易所恢复 {_iso()}"


def format_error(detail: str) -> str:
    return f"🔴 程序报错 {_iso()}\n{detail[:1500]}"


def format_halt(reason: str, equity: float) -> str:
    return f"⛔ 风控停机 {_iso()}\n原因 {_zh_halt(reason)}\n权益 {equity:.2f}"


def _pnl_txt(pnl: float, pnl_pct: float | None = None) -> str:
    word = "盈利" if pnl > 0 else ("亏损" if pnl < 0 else "盈亏")
    pct_s = f"（{_fmt_pct(pnl_pct)}）" if pnl_pct is not None else ""
    return f"{word} {pnl:+.2f}{pct_s}"


def format_fill(order: dict) -> str:
    sym = order.get("symbol")
    action = _zh_action(str(order.get("action") or ""))
    side = _zh_side(str(order.get("side") or ""))
    qty = order.get("qty")
    px = float(order.get("avg_price") or 0)
    px_s = _fmt_px(px) if px else "市价"
    bar = order.get("bar") or "—"
    st = _zh_status(str(order.get("status") or ""))
    lines = [f"📋 成交 {sym} {action} {side} {qty} @ {px_s}"]
    pnl = order.get("pnl")
    if str(order.get("action") or "").upper() == "CLOSE" and pnl is not None:
        lines.append(_pnl_txt(float(pnl), order.get("pnl_pct")))
    lines.append(f"K线 {bar}  状态 {st}")
    return "\n".join(lines)


def format_fit_warn(leg: dict) -> str:
    return (
        f"⚠️ 形态走样预警 {leg.get('symbol')} {_iso()}\n"
        f"{format_leg(leg).strip()}\n"
        "拟合低于 20%，接近出场阈值 0%"
    )


class TelegramNotifier:
    def __init__(self, reports_dir: Path, loop_min: float = 5.0):
        load_env_file()
        self.token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
        self.chat_id = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
        self.enabled = bool(self.token and self.chat_id)
        self.path = Path(reports_dir) / STATE_NAME
        self.loop_min = loop_min
        self.host = socket.gethostname()
        self.state = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                pass
        return {"last": {}, "disconnected": False, "halted": False, "fit_warn": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=float))
        tmp.replace(self.path)

    def _cooled(self, key: str, cooldown_s: float) -> bool:
        if cooldown_s <= 0:
            return False
        last = float((self.state.get("last") or {}).get(key) or 0)
        return (time.time() - last) < cooldown_s

    def _touch(self, key: str) -> None:
        self.state.setdefault("last", {})[key] = time.time()
        self._save()

    def send(self, text: str, key: str | None = None, cooldown_s: float = 0.0) -> bool:
        if not self.enabled or not text:
            return False
        if key and self._cooled(key, cooldown_s):
            return False
        body = text if len(text) <= MAX_LEN else text[: MAX_LEN - 20] + "\n…(截断)"
        try:
            r = requests.post(
                TG_API.format(token=self.token),
                json={
                    "chat_id": self.chat_id,
                    "text": body,
                    "disable_web_page_preview": True,
                },
                timeout=8,
            )
            if r.status_code >= 400:
                print(f"telegram {r.status_code}: {r.text[:200]}", flush=True)
                return False
        except Exception as e:
            print(f"telegram send failed: {e}", flush=True)
            return False
        if key:
            self._touch(key)
        return True

    def due_hourly(self) -> bool:
        last = float((self.state.get("last") or {}).get("hourly") or 0)
        return (time.time() - last) >= 3600.0

    def online(self, snap: dict) -> None:
        snap = dict(snap)
        snap.setdefault("host", self.host)
        snap.setdefault("loop_min", self.loop_min)
        self.send(format_online(snap), key="online", cooldown_s=5)

    def offline(self, reason: str = "") -> None:
        self.send(format_offline(reason), key="offline", cooldown_s=5)

    def disconnect(self, detail: str) -> None:
        if self.state.get("disconnected"):
            return
        if self.send(format_disconnect(detail), key="disconnect", cooldown_s=60):
            self.state["disconnected"] = True
            self._save()

    def reconnect(self) -> None:
        if not self.state.get("disconnected"):
            return
        self.state["disconnected"] = False
        self._save()
        self.send(format_reconnect(), key="reconnect", cooldown_s=5)

    def error(self, detail: str) -> None:
        self.send(format_error(detail), key=f"err:{detail[:80]}", cooldown_s=900)

    def halt(self, reason: str, equity: float) -> None:
        if self.state.get("halted"):
            return
        if self.send(format_halt(reason, equity)):
            self.state["halted"] = True
            self._save()

    def clear_halt(self) -> None:
        if self.state.get("halted"):
            self.state["halted"] = False
            self._save()

    def fills(self, orders: list[dict]) -> None:
        for o in orders:
            self.send(format_fill(o))

    def fit_warn(self, leg: dict) -> None:
        fit = leg.get("fit")
        sym = str(leg.get("symbol") or "")
        if fit is None or fit >= FIT_WARN or int(leg.get("pos") or 0) == 0:
            if sym:
                self.state.setdefault("fit_warn", {}).pop(sym, None)
                self._save()
            return
        warned = bool((self.state.get("fit_warn") or {}).get(sym))
        if warned:
            return
        if self.send(format_fit_warn(leg), key=f"fit:{sym}", cooldown_s=21600):
            self.state.setdefault("fit_warn", {})[sym] = True
            self._save()

    def hourly(self, snap: dict) -> None:
        snap = dict(snap)
        snap.setdefault("host", self.host)
        snap.setdefault("loop_min", self.loop_min)
        if self.send(format_hourly(snap), key="hourly", cooldown_s=3300):
            return

    def stale_bar(self, last_bar: str | None, hours: float) -> None:
        if hours < 2.2:
            return
        self.send(
            f"🟠 步进落后 {_iso()}\n上次K线 {last_bar}\n已超过 {hours:.1f} 小时未处理新K线",
            key="stale_bar",
            cooldown_s=3600,
        )
