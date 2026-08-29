"""Binance USDT-M broker. Keys from env only. Testnet only.

Env:
  BINANCE_TESTNET_API_KEY
  BINANCE_TESTNET_API_SECRET
"""
from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

TESTNET_FAPI = "https://testnet.binancefuture.com"
LIVE_FAPI = "https://fapi.binance.com"
_ENV_LOADED = False

# Transient server-side failures worth retrying: HTTP 5xx and Binance error
# codes like -1000/-1001 (internal errors). Retries are same-request (same
# timestamp/query) so a retry never double-fires an order that already landed.
_TRANSIENT_HTTP_RE = re.compile(r"binance 5\d\d:")
_TRANSIENT_CODES = {"-1000", "-1001"}
_MAX_RETRIES = 2
_RETRY_SLEEP_S = 2.0


def _is_transient_error(exc: Exception) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc)
    if _TRANSIENT_HTTP_RE.search(msg):
        return True
    return any(f'"code":{c}' in msg for c in _TRANSIENT_CODES)


def load_env_file(path: Path | None = None) -> Path | None:
    """Load KEY=VALUE from a local .env if env vars are unset. Never overrides."""
    global _ENV_LOADED
    if path is None:
        path = Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return None
    if _ENV_LOADED and path == Path(__file__).resolve().parents[1] / ".env":
        return path
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val
    if path == Path(__file__).resolve().parents[1] / ".env":
        _ENV_LOADED = True
    return path
KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


class BinanceUSDMBroker:
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        timeout: float = 12.0,
    ):
        load_env_file()
        self.api_key = api_key or os.environ.get("BINANCE_TESTNET_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("BINANCE_TESTNET_API_SECRET", "")
        self.testnet = testnet
        self.timeout = timeout
        self.base = TESTNET_FAPI if testnet else LIVE_FAPI
        if not testnet:
            raise RuntimeError("live trading is disabled in this broker")
        if not self.api_key or not self.api_secret:
            raise RuntimeError("set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET")
        self._filters: dict | None = None

    def ping(self) -> dict:
        r = requests.get(self.base + "/fapi/v1/ping", timeout=self.timeout)
        r.raise_for_status()
        return {"ok": True, "base": self.base, "http": r.status_code}

    def server_time(self) -> int:
        r = requests.get(self.base + "/fapi/v1/time", timeout=self.timeout)
        r.raise_for_status()
        return int(r.json()["serverTime"])

    def _public(self, path: str, params: dict | None = None):
        r = requests.get(self.base + path, params=params or {}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _signed(self, method: str, path: str, params: dict) -> dict:
        q = dict(params)
        q["timestamp"] = int(time.time() * 1000)
        q["recvWindow"] = 60000
        query = urlencode(q, doseq=True)
        sig = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"{self.base}{path}?{query}&signature={sig}"
        headers = {"X-MBX-APIKEY": self.api_key}
        for attempt in range(_MAX_RETRIES + 1):
            r = requests.request(method, url, headers=headers, timeout=self.timeout)
            if r.status_code < 400:
                return r.json() if r.text else {}
            err = RuntimeError(f"binance {r.status_code}: {r.text}")
            if attempt < _MAX_RETRIES and _is_transient_error(err):
                time.sleep(_RETRY_SLEEP_S)
                continue
            raise err

    def symbol_filters(self, symbol: str) -> dict:
        if self._filters is None:
            info = self._public("/fapi/v1/exchangeInfo")
            self._filters = {}
            for s in info.get("symbols") or []:
                self._filters[s["symbol"]] = s
        return self._filters.get(symbol) or {}

    def round_qty(self, symbol: str, qty: float) -> float:
        meta = self.symbol_filters(symbol)
        step = 0.0001
        min_qty = 0.0001
        for f in meta.get("filters") or []:
            if f.get("filterType") == "LOT_SIZE":
                step = float(f.get("stepSize") or step)
                min_qty = float(f.get("minQty") or min_qty)
        if qty <= 0:
            return 0.0
        n = math.floor(qty / step + 1e-12) * step
        prec = meta.get("quantityPrecision")
        if prec is not None:
            n = float(f"{n:.{int(prec)}f}")
        if n < min_qty:
            return 0.0
        return n

    def min_notional(self, symbol: str) -> float:
        meta = self.symbol_filters(symbol)
        for f in meta.get("filters") or []:
            if f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL"):
                return float(f.get("notional") or f.get("minNotional") or 0)
        return 0.0

    def mark_price(self, symbol: str) -> float:
        row = self._public("/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(row["markPrice"])

    def klines(self, symbol: str, interval: str, limit: int, closed_only: bool = True) -> pd.DataFrame:
        raw = self._public(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": min(1500, max(1, limit))},
        )
        df = pd.DataFrame(raw, columns=KLINE_COLS)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        if closed_only:
            now = pd.Timestamp.now(tz="UTC")
            df = df[df["close_time"] < now]
        return df.reset_index(drop=True)

    def account(self) -> dict:
        return self._signed("GET", "/fapi/v2/account", {})

    def usdt_balance(self) -> dict:
        acct = self.account()
        assets = acct.get("assets") or []
        usdt = next((a for a in assets if a.get("asset") == "USDT"), {})
        wallet = float(usdt.get("walletBalance") or acct.get("totalWalletBalance") or 0)
        unreal = float(usdt.get("unrealizedProfit") or acct.get("totalUnrealizedProfit") or 0)
        equity = float(acct.get("totalMarginBalance") or usdt.get("marginBalance") or (wallet + unreal))
        return {
            "wallet": wallet,
            "unrealized": unreal,
            "equity": equity,
            "available": float(usdt.get("availableBalance") or 0),
            "can_trade": bool(acct.get("canTrade")),
        }

    def position(self, symbol: str) -> dict:
        rows = self._signed("GET", "/fapi/v2/positionRisk", {"symbol": symbol})
        if isinstance(rows, list) and rows:
            return rows[0]
        return {}

    def position_amt(self, symbol: str) -> float:
        return float(self.position(symbol).get("positionAmt") or 0)

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        return self._signed("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": int(leverage)})

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> dict:
        try:
            return self._signed("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type})
        except RuntimeError as e:
            if "No need to change" in str(e) or '"code":-4046' in str(e):
                return {"ok": True, "unchanged": True}
            raise

    def market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        client_order_id: str = "",
    ) -> dict:
        q = self.round_qty(symbol, qty)
        if q <= 0:
            raise RuntimeError(f"qty rounded to 0 for {symbol} raw={qty}")
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": f"{q:.8f}".rstrip("0").rstrip("."),
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if client_order_id:
            params["newClientOrderId"] = client_order_id[:36]
        return self._signed("POST", "/fapi/v1/order", params)


def probe_testnet(timeout: float = 8.0) -> dict:
    url = TESTNET_FAPI + "/fapi/v1/ping"
    try:
        r = requests.get(url, timeout=timeout)
        return {"reachable": r.status_code == 200, "http": r.status_code, "url": url}
    except Exception as e:
        return {"reachable": False, "http": 0, "url": url, "error": str(e)}
