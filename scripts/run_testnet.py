#!/usr/bin/env python3
"""Binance USDT-M *testnet* live step. Real orders. Keys from env only.

  export BINANCE_TESTNET_API_KEY=...
  export BINANCE_TESTNET_API_SECRET=...
  python scripts/run_testnet.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live.broker_binance import BinanceUSDMBroker, load_env_file, probe_testnet  # noqa: E402
from live.notify import TelegramNotifier  # noqa: E402
from live.runloop import RunLoop  # noqa: E402
from research.config import PassedConfig  # noqa: E402
from research.log import get_logger  # noqa: E402

log = get_logger("testnet")

REPORT_DIR = ROOT / "reports"


def _make_broker() -> BinanceUSDMBroker:
    return BinanceUSDMBroker(testnet=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loop", type=float, default=0.0, help="minutes between steps; 0 = single run")
    p.add_argument("--min-notional", type=float, default=55.0, help="skip opens below this USD notional")
    args = p.parse_args()

    load_env_file()
    cfg = PassedConfig.load(ROOT / "configs" / "passed.yaml")
    tg = TelegramNotifier(REPORT_DIR, loop_min=args.loop or 5.0)

    # Validate testnet connectivity
    ping = probe_testnet()
    if not ping.get("reachable"):
        log.warning("testnet unreachable from this network")
    else:
        log.info("testnet reachable: %s", ping.get("url"))

    loop = RunLoop(
        config=cfg,
        report_dir=REPORT_DIR,
        broker_factory=_make_broker,
        tg=tg,
        min_notional=args.min_notional,
    )

    if args.loop <= 0:
        loop.run_all()
        return

    loop.run_loop(args.loop)


if __name__ == "__main__":
    main()
