#!/usr/bin/env python3
"""Check Binance USDT-M testnet reachability and positions. No orders."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live.broker_binance import BinanceUSDMBroker, load_env_file, probe_testnet  # noqa: E402


def main() -> None:
    load_env_file()
    ping = probe_testnet()
    out = {
        "ping": ping,
        "keys": bool(os.environ.get("BINANCE_TESTNET_API_KEY") and os.environ.get("BINANCE_TESTNET_API_SECRET")),
        "positions": {},
    }
    if ping.get("reachable") and out["keys"]:
        try:
            b = BinanceUSDMBroker(testnet=True)
            out["server_time"] = b.server_time()
            out["balance"] = b.usdt_balance()
            out["account_ok"] = True
            for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
                pos = b.position(symbol)
                out["positions"][symbol] = {
                    "amt": float(pos.get("positionAmt") or 0),
                    "entry": float(pos.get("entryPrice") or 0),
                    "mark": float(pos.get("markPrice") or 0),
                    "unrealized": float(pos.get("unRealizedProfit") or 0),
                    "leverage": pos.get("leverage"),
                    "margin_type": pos.get("marginType"),
                }
        except Exception as e:
            out["account_ok"] = False
            out["account_error"] = str(e)
    print(json.dumps(out, indent=2, default=float))
    if not ping.get("reachable"):
        print("TESTNET UNREACHABLE from this network — keep local simulated fills")
        sys.exit(2)
    if not out["keys"]:
        print("NO KEYS — set BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET or put them in .env")
        sys.exit(3)


if __name__ == "__main__":
    main()
