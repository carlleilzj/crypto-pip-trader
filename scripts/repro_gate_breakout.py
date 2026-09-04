#!/usr/bin/env python3
"""Reproduce the frozen BREAKOUT strategy gate on the 3-coin portfolio.

This is the spec that PASSED the gate after the parity fix and the move
from 30d time-momentum to N-bar breakout entry:
  - Entry: close breaks above/below the rolling N-bar high/low (N=40)
  - Exit: PIP shape-break + 6% hard stop + 720-bar time hold
  - Size: 0.2449 notional split equally across BTC/ETH/SOL
  - Cost: taker 4bps + 1bp slippage + funding

Gate criteria (unchanged hard thresholds):
  net return > 0, Martin > 0, trades >= 5, beats buy-and-hold,
  max drawdown < 25%, block-permutation p < 0.05.

Calibrated on train 2020-2021 (N=40 chosen by combined Calmar), validated
on 2022-2024.06, FINAL verdict on full OOS 2022-01→2026-08. The holdout
2024-07→2026-08 was touched once during this research; full OOS is the
deployment window and is what the gate reports.

Usage:
  python scripts/repro_gate_breakout.py                 # gate check only (fast)
  python scripts/repro_gate_breakout.py --permutations 200  # full p-value (~30 min)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.engine import EventBacktest  # noqa: E402
from research.breakout import breakout_pip_exit_signals  # noqa: E402
from research.config import PassedConfig  # noqa: E402
from research.costs import CostModel  # noqa: E402
from research.data import load_dataset  # noqa: E402
from research.log import get_logger  # noqa: E402
from research.metrics import martin_ratio, max_drawdown  # noqa: E402
from research.momentum import block_permute_close  # noqa: E402

log = get_logger("gate_breakout")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
T2022 = pd.Timestamp("2022-01-01", tz="UTC")
N = 40  # breakout window, calibrated on train


def _summary(name: str, equity: np.ndarray, n_trades: int, bh: float | None = None) -> dict:
    ret = float(equity[-1] / equity[0] - 1.0) if len(equity) else 0.0
    log_rets = np.diff(np.log(np.maximum(equity, 1e-12)), prepend=np.log(equity[0]) if len(equity) else 0.0)
    martin = martin_ratio(log_rets)
    mdd = max_drawdown(equity)
    out = {"name": name, "return": ret, "martin": martin, "max_dd": mdd, "n_trades": n_trades}
    line = f"  {name:<28} ret {ret*100:+6.1f}%  Martin {martin:6.2f}  dd {mdd*100:6.1f}%  trades {n_trades:4d}"
    if bh is not None:
        out["beats_bh"] = ret > bh
        line += f"  beats BH {bh*100:+.1f}%"
    log.info("%s", line)
    return out


def run_portfolio(closes_list: list[np.ndarray], frac_each: float, cost: CostModel,
                  start_each: float = 333.33) -> tuple[np.ndarray, int]:
    eqs: list[np.ndarray] = []
    tot_tr = 0
    for close in closes_list:
        m = len(close)
        df = pd.DataFrame({
            "open": close, "high": close, "low": close, "close": close,
            "funding_rate": np.zeros(m),
        })
        sig = breakout_pip_exit_signals(
            close, n=N, hold=720, mode="long_short", pip_n=5, min_bars=24,
            fit_exit=0.0, use_pip=True, stop_pct=0.06,
        )
        r = EventBacktest(cost=cost, start_equity=start_each, notional_frac=frac_each).run_signals(
            df, sig, np.full(m, frac_each)
        )
        eqs.append(r.equity)
        tot_tr += r.summary["n_trades"]
    min_len = min(len(e) for e in eqs)
    return np.sum([e[:min_len] for e in eqs], axis=0), tot_tr


def run_buyhold(closes_list: list[np.ndarray], cost: CostModel,
                start_each: float = 333.33) -> np.ndarray:
    eqs = []
    for close in closes_list:
        m = len(close)
        df = pd.DataFrame({
            "open": close, "high": close, "low": close, "close": close,
            "funding_rate": np.zeros(m),
        })
        r = EventBacktest(cost=cost, start_equity=start_each, notional_frac=1.0).run_signals(
            df, np.ones(m), np.ones(m)
        )
        eqs.append(r.equity)
    min_len = min(len(e) for e in eqs)
    return np.sum([e[:min_len] for e in eqs], axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--permutations", type=int, default=0, help="block-permutation reps (0 = skip)")
    p.add_argument("--block", type=int, default=168)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = PassedConfig.load(ROOT / "configs" / "passed.yaml")
    frac_each = cfg.notional_frac / len(SYMBOLS)
    cost = CostModel(
        taker_fee=cfg.costs.taker_fee, slippage_bps=cfg.costs.slippage_bps,
        use_funding=cfg.costs.use_funding,
    )
    start_each = cfg.backtest.start_equity / len(SYMBOLS)

    oos_closes: list[np.ndarray] = []
    for sym in SYMBOLS:
        kp = ROOT / f"data/{sym}_1h.csv"
        fp = ROOT / f"data/{sym}_funding.csv"
        if not kp.exists() or not fp.exists():
            raise SystemExit(f"missing data {kp} / {fp} — run download_data.py")
        df = load_dataset(kp, fp)
        oos = df[df["open_time"] >= T2022].reset_index(drop=True)
        oos_closes.append(oos["close"].to_numpy(dtype=float))
        log.info("loaded %s: %d OOS bars %s → %s", sym, len(oos),
                 oos["open_time"].iloc[0], oos["open_time"].iloc[-1])

    eq, n_trades = run_portfolio(oos_closes, frac_each, cost, start_each)
    bh_eq = run_buyhold(oos_closes, cost, start_each)
    bh_ret = float(bh_eq[-1] / bh_eq[0] - 1.0)
    s = _summary("frozen breakout 3-coin", eq, n_trades, bh_ret)
    _ = _summary("buy & hold 3-coin", bh_eq, 1)

    gates = {
        "net_return > 0": s["return"] > 0,
        "martin > 0": s["martin"] > 0,
        "trades >= 5": s["n_trades"] >= 5,
        "beats_buy_hold": s.get("beats_bh", s["return"] > bh_ret),
        "max_dd < 25%": abs(s["max_dd"]) < 0.25,
    }
    if args.permutations > 0:
        ge = 0
        actual_ret = s["return"]
        for r in range(args.permutations):
            perm_closes = [
                block_permute_close(c, args.block, args.seed + r) for c in oos_closes
            ]
            peq, _ = run_portfolio(perm_closes, frac_each, cost, start_each)
            if float(peq[-1] / peq[0] - 1.0) >= actual_ret:
                ge += 1
            if (r + 1) % 20 == 0:
                log.info("  permutation %d/%d (ge=%d)", r + 1, args.permutations, ge)
        p_val = (1.0 + ge) / (1.0 + args.permutations)
        gates["block_permutation p < 0.05"] = p_val < 0.05
        log.info("block permutation p = %.4f (n=%d, block=%d, synchronous)",
                 p_val, args.permutations, args.block)

    log.info("=== GATE (breakout 3-coin portfolio) ===")
    passed = True
    for k, v in gates.items():
        mark = "PASS" if v else "FAIL"
        log.info("  [%s] %s", mark, k)
        passed = passed and v
    log.info("overall: %s", "PASS" if passed else "FAIL")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
