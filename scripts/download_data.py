#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.data import download_funding, download_klines  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    k = download_klines(
        cfg["symbol"],
        cfg["interval"],
        cfg["data"]["start"],
        ROOT / cfg["data"]["kline_path"],
    )
    f = download_funding(
        cfg["symbol"],
        cfg["data"]["start"],
        ROOT / cfg["data"]["funding_path"],
    )
    print(f"klines={len(k)} {k['open_time'].iloc[0]} -> {k['open_time'].iloc[-1]}")
    print(f"funding={len(f)} {f['fundingTime'].iloc[0]} -> {f['fundingTime'].iloc[-1]}")


if __name__ == "__main__":
    main()
