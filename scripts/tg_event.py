#!/usr/bin/env python3
"""Send a one-off Telegram event. Used by systemd OnFailure and manual tests.

  python scripts/tg_event.py --test
  python scripts/tg_event.py --event crash
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live.notify import TelegramNotifier, format_error, format_offline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    p.add_argument("--event", choices=["crash", "offline"], default="")
    args = p.parse_args()
    tg = TelegramNotifier(ROOT / "reports")
    if not tg.enabled:
        print("未配置 Telegram：请设置 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID", flush=True)
        sys.exit(2)
    if args.test:
        ok = tg.send(f"🧪 盯仓通知测试\n主机 {tg.host}")
        sys.exit(0 if ok else 1)
    if args.event == "crash":
        ok = tg.send(format_error("盯仓进程异常退出，系统正在自动重启"))
        sys.exit(0 if ok else 1)
    if args.event == "offline":
        ok = tg.send(format_offline("服务停止"))
        sys.exit(0 if ok else 1)
    p.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
