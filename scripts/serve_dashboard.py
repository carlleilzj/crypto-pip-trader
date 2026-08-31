#!/usr/bin/env python3
"""Local dashboard. No auth, bind localhost only.

  python scripts/serve_dashboard.py
  open http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.pips import find_pips  # noqa: E402

REPORTS = ROOT / "reports"
HOST = "127.0.0.1"
PORT = 8765
TESTNET_KLINES = "https://testnet.binancefuture.com/fapi/v1/klines"
PUBLIC_SPOT = "https://data-api.binance.vision/api/v3/klines"
HTML_PATH = ROOT / "scripts" / "dashboard.html"


def _read_json(path: Path, default=None):
    if not path.exists():
        return default if default is not None else {}
    return json.loads(path.read_text())


_KLINE_CACHE: dict[tuple[str, int], tuple[float, tuple[list[dict], str]]] = {}
_KLINE_TTL_S = 60.0


def _fetch_klines(symbol: str = "BTCUSDT", limit: int = 96) -> tuple[list[dict], str]:
    # Browser auto-refresh used to hammer testnet klines and trip the IP
    # auto-ban (HTTP 418) that also throttles the trading loop — cache 60s.
    import time as _time

    key = (symbol, limit)
    cached = _KLINE_CACHE.get(key)
    if cached and (_time.time() - cached[0]) < _KLINE_TTL_S:
        return cached[1]
    params = {"symbol": symbol, "interval": "1h", "limit": limit}
    try:
        r = requests.get(TESTNET_KLINES, params=params, timeout=8)
        r.raise_for_status()
        raw = r.json()
        src = "BINANCE TESTNET"
    except Exception:
        r = requests.get(PUBLIC_SPOT, params=params, timeout=8)
        r.raise_for_status()
        raw = r.json()
        src = "SPOT PUBLIC"
    bars = []
    for row in raw:
        bars.append(
            {
                "t": int(row[0]),
                "o": float(row[1]),
                "h": float(row[2]),
                "l": float(row[3]),
                "c": float(row[4]),
            }
        )
    _KLINE_CACHE[key] = (_time.time(), (bars, src))
    return bars, src


def _pips_on(closes: np.ndarray, start: int, end: int, n: int = 5) -> list[dict]:
    if end <= start:
        return []
    window = closes[start:end]
    k = min(n, len(window))
    if k < 2:
        return [{"i": int(start + i), "y": float(window[i])} for i in range(len(window))]
    px, py = find_pips(window, k, 3)
    return [{"i": int(start + xi), "y": float(yi)} for xi, yi in zip(px, py, strict=False)]


def _z(ys: list[float]) -> np.ndarray:
    a = np.asarray(ys, dtype=float)
    std = float(a.std())
    if std < 1e-12:
        return np.zeros_like(a)
    return (a - a.mean()) / std


def _project_shape(pred: list[dict], actual: list[dict]) -> list[dict]:
    if len(pred) < 2 or len(actual) < 2:
        return []
    ay = np.array([p["y"] for p in actual], dtype=float)
    mean, std = float(ay.mean()), float(ay.std())
    if std < 1e-12:
        return []
    zs = _z([p["y"] for p in pred])
    return [{"i": actual[i]["i"], "y": float(mean + zs[i] * std)} for i in range(min(len(pred), len(actual)))]


def _shape_fit(pred: list[dict], actual: list[dict]) -> float | None:
    if len(pred) < 3 or len(actual) < 3:
        return None
    n = min(len(pred), len(actual))
    a = _z([p["y"] for p in pred][:n])
    b = _z([p["y"] for p in actual][:n])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def _state_for(symbol: str) -> dict:
    p = REPORTS / "testnet" / f"{symbol}_state.json"
    if p.exists():
        return _read_json(p, {})
    if symbol == "BTCUSDT":
        return _read_json(REPORTS / "testnet_state.json", {})
    return {}


def chart_payload(symbol: str = "BTCUSDT") -> dict:
    symbol = (symbol or "BTCUSDT").upper()
    if symbol not in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        symbol = "BTCUSDT"
    bars, src = _fetch_klines(symbol, 180)
    closes = np.array([b["c"] for b in bars], dtype=float)
    times = [int(b["t"]) for b in bars]
    tn = _state_for(symbol)
    look = 24
    pred, actual, ghost = [], [], []
    fit = None
    entry_i = None
    last_bar = tn.get("last_bar")
    if last_bar and times:
        ts = int(pd.Timestamp(last_bar).timestamp() * 1000)
        for i, t in enumerate(times):
            if t >= ts:
                entry_i = i
                break
        if entry_i is None:
            entry_i = len(times) - 1
        frozen = tn.get("pred_pips")
        if frozen:
            px = [int(p["i"]) for p in frozen]
            py = np.array([float(p["y"]) for p in frozen])
            entry_px = float(tn.get("entry") or closes[entry_i])
            base = py[0] if py[0] else 1.0
            scale = entry_px / base
            pred = [{"i": entry_i + xi, "y": float(y * scale)} for xi, y in zip(px, py, strict=False)]
            actual = _pips_on(closes, entry_i, len(closes), 5)
            fit = _shape_fit(pred, actual)
        else:
            pred_start = max(0, entry_i - look + 1)
            pred = _pips_on(closes, pred_start, entry_i + 1, 5)
            actual = _pips_on(closes, entry_i, len(closes), 5)
            fit = _shape_fit(pred, actual)
        last_pred_i = max((p["i"] for p in pred), default=entry_i)
        if last_pred_i >= len(bars) and bars:
            step = 3600_000
            last_t = int(bars[-1]["t"])
            last_c = float(bars[-1]["c"])
            for k in range(1, last_pred_i - len(bars) + 2):
                bars.append({
                    "t": last_t + k * step, "o": last_c, "h": last_c,
                    "l": last_c, "c": last_c, "ghost": True,
                })
    elif len(closes) >= 5:
        actual = _pips_on(closes, max(0, len(closes) - look), len(closes), 5)
    asof = ""
    if bars:
        asof = pd.to_datetime(bars[-1]["t"], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M UTC")
    hold_left = int(tn.get("hold_left") or 0)
    pos = int(tn.get("pos") or 0)
    marks = []
    exit_at = ""
    if entry_i is not None and pos != 0:
        marks.append({"kind": "entry", "i": int(entry_i), "side": pos})
        if tn.get("exit_mode", "pip") != "time":
            exit_at = "shape fit < 0"
            if fit is not None and fit < 0:
                marks.append({"kind": "exit", "i": len(closes) - 1, "side": -pos, "planned": False})
        else:
            exit_i = entry_i + max(hold_left, 0)
            if last_bar:
                exit_ts = pd.Timestamp(last_bar, tz="UTC") + pd.Timedelta(hours=max(hold_left, 0))
                exit_at = exit_ts.strftime("%m-%d %H:%M UTC")
            pad = 12
            vis_exit = min(exit_i, len(bars) - 1 + pad)
            if vis_exit >= len(bars) and bars:
                step = 3600_000
                last_t = int(bars[-1]["t"])
                last_c = float(bars[-1]["c"])
                for k in range(1, vis_exit - len(bars) + 2):
                    bars.append({
                        "t": last_t + k * step, "o": last_c, "h": last_c,
                        "l": last_c, "c": last_c, "ghost": True,
                    })
            marks.append({"kind": "exit", "i": int(min(exit_i, len(bars) - 1)), "side": -pos, "planned": True})
    return {
        "bars": bars,
        "pred": pred,
        "actual": actual,
        "ghost": ghost,
        "fit": fit,
        "future_from": len(closes),
        "marks": marks,
        "hold_left": hold_left if pos else None,
        "exit_at": exit_at,
        "bars_in_trade": (len(closes) - entry_i) if entry_i is not None else 0,
        "min_bars": 24,
        "entry": float(tn["entry"]) if tn.get("entry") else None,
        "pos": pos,
        "qty": float(tn.get("qty") or 0),
        "equity": tn.get("equity"),
        "symbol": symbol,
        "source": src,
        "asof": asof,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = HTML_PATH.read_text(encoding="utf-8") if HTML_PATH.exists() else "<h1>dashboard.html not found</h1>"
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path in ("/api/chart", "/api/dashboard"):
            q = urlparse(self.path).query
            from urllib.parse import parse_qs
            symbol = (parse_qs(q).get("symbol") or ["BTCUSDT"])[0]
            self._send(200, json.dumps(chart_payload(symbol), default=float).encode(), "application/json")
            return
        self._send(404, b"not found", "text/plain")


def main() -> None:
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"dashboard http://{HOST}:{PORT}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
