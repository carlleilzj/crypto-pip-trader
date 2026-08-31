#!/usr/bin/env python3
"""Reproduce the frozen strategy gate (single-coin BTC, 30d momentum).

Runs the exact spec in configs/passed.yaml against the BTC dataset and
checks every hard gate criterion from the README:

  net return > 0, Martin > 0, trades >= 5, beats buy-and-hold,
  max drawdown < 25%, block-permutation p < 0.05.

Also prints the comparison rows from the README "已跑对照" table so the
frozen spec can be audited against alternatives without re-tuning on OOS.

Usage:
  python scripts/repro_gate.py                 # gate check only (fast)
  python scripts/repro_gate.py --permutations 1000  # full p-value (slower)
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
from research.config import PassedConfig  # noqa: E402
from research.costs import CostModel  # noqa: E402
from research.data import load_dataset  # noqa: E402
from research.log import get_logger  # noqa: E402
from research.metrics import martin_ratio, max_drawdown  # noqa: E402
from research.momentum import (  # noqa: E402
    block_permute_close,
    calibrate_notional,
    constant_sizes,
    oos_slice,
    ts_momentum_pip_exit,
    ts_momentum_signals,
)

log = get_logger("gate")

KL = ROOT / "data" / "BTCUSDT_1h.csv"
FU = ROOT / "data" / "BTCUSDT_funding.csv"
TRAIN_H = 17520  # 2020-01-01 → 2021-12-31 (calibration window)


def _summary(name: str, equity: np.ndarray, trades: pd.DataFrame, bh: float | None = None) -> dict:
    ret = float(equity[-1] / equity[0] - 1.0) if len(equity) else 0.0
    log_eq = np.log(np.maximum(equity, 1e-12))
    log_rets = np.diff(log_eq, prepend=log_eq[0])
    martin = martin_ratio(log_rets)
    mdd = max_drawdown(equity)
    n = int(len(trades))
    out = {"name": name, "return": ret, "martin": martin, "max_dd": mdd, "n_trades": n}
    line = f"  {name:<28} ret {ret*100:+6.1f}%  Martin {martin:6.2f}  dd {mdd*100:6.1f}%  trades {n:3d}"
    if bh is not None:
        out["beats_bh"] = ret > bh
        line += f"  beats BH {bh*100:+.1f}%"
    log.info("%s", line)
    return out


def run_frozen(df: pd.DataFrame, cfg: PassedConfig) -> tuple[np.ndarray, pd.DataFrame]:
    """Run the frozen 30d-momentum + PIP-exit strategy on `df`."""
    close = df["close"].to_numpy(dtype=float)
    sig = ts_momentum_pip_exit(
        close,
        lookback=cfg.lookback,
        hold=cfg.hold,
        mode=cfg.mode,
        pip_n=cfg.pip_n,
        min_bars=cfg.min_bars,
        fit_exit=cfg.fit_exit,
        use_pip=(cfg.exit_mode == "pip"),
        stop_pct=cfg.stop_pct,
    )
    sizes = constant_sizes(len(df), cfg.notional_frac)
    cost = CostModel(
        taker_fee=cfg.costs.taker_fee,
        slippage_bps=cfg.costs.slippage_bps,
        use_funding=cfg.costs.use_funding,
    )
    bt = EventBacktest(cost=cost, start_equity=cfg.backtest.start_equity, notional_frac=cfg.notional_frac)
    res = bt.run_signals(df, sig, sizes)
    return res.equity, res.trades


def run_baseline(df: pd.DataFrame, lookback: int, hold: int, frac: float, cfg: PassedConfig) -> np.ndarray:
    """Run a plain time-momentum baseline (no PIP exit, no stop)."""
    close = df["close"].to_numpy(dtype=float)
    sig = ts_momentum_signals(close, lookback=lookback, hold=hold, mode=cfg.mode)
    sizes = constant_sizes(len(df), frac)
    cost = CostModel(
        taker_fee=cfg.costs.taker_fee,
        slippage_bps=cfg.costs.slippage_bps,
        use_funding=cfg.costs.use_funding,
    )
    bt = EventBacktest(cost=cost, start_equity=cfg.backtest.start_equity, notional_frac=frac)
    return bt.run_signals(df, sig, sizes).equity


def permutation_p(
    close: np.ndarray, actual_ret: float, cfg: PassedConfig,
    n_reps: int, block: int, seed0: int,
) -> float:
    """Block-permutation p-value: fraction of shuffled paths beating actual.

    Uses the same frozen strategy on each shuffled path so the test is
    apples-to-apples. p = (1 + #{shuffled >= actual}) / (1 + n_reps).
    """
    ge = 0
    for r in range(n_reps):
        perm = block_permute_close(close, block, seed0 + r)
        df_p = pd.DataFrame({"open": perm, "high": perm, "low": perm, "close": perm, "funding_rate": 0.0})
        eq, _ = run_frozen(df_p, cfg)
        if float(eq[-1] / eq[0] - 1.0) >= actual_ret:
            ge += 1
        if (r + 1) % 50 == 0:
            log.info("  permutation %d/%d (ge=%d)", r + 1, n_reps, ge)
    return (1.0 + ge) / (1.0 + n_reps)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--permutations", type=int, default=0, help="block-permutation reps (0 = skip)")
    p.add_argument("--block", type=int, default=168, help="permutation block size (hours)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = PassedConfig.load(ROOT / "configs" / "passed.yaml")
    if not KL.exists() or not FU.exists():
        raise SystemExit(
            f"missing data {KL} / {FU} — run `python scripts/download_data.py` first"
        )
    df = load_dataset(KL, FU)
    log.info("loaded %d bars %s → %s", len(df), df["open_time"].iloc[0], df["open_time"].iloc[-1])

    # Calibrate notional from train window only (never OOS).
    train = df.iloc[:TRAIN_H]
    train_eq, _ = run_frozen(train, cfg)
    train_dd = max_drawdown(train_eq)
    calibrated = calibrate_notional(train_dd, base_frac=0.5, target_dd=0.20)
    log.info("train dd %.1f%% → calibrated frac %.4f (frozen: %.4f)", train_dd * 100, calibrated, cfg.notional_frac)

    # OOS slice.
    oos = oos_slice(df, TRAIN_H)
    log.info("OOS %d bars %s → %s", len(oos), oos["open_time"].iloc[0], oos["open_time"].iloc[-1])

    # Frozen strategy on OOS.
    eq, trades = run_frozen(oos, cfg)
    bh_eq = run_baseline(oos, lookback=1, hold=len(oos), frac=1.0, cfg=cfg)  # buy & hold = always long, full size
    bh_ret = float(bh_eq[-1] / bh_eq[0] - 1.0)
    s = _summary("frozen 30d mom + pip exit", eq, trades, bh_ret)
    _ = _summary("buy & hold", bh_eq, pd.DataFrame({"pnl": [0.0]}))

    # README comparison rows (alternatives, for transparency — not gates).
    log.info("comparison rows (README 已跑对照):")
    for lb, hd, frac, label in [
        (168, 168, 0.5, "7d momentum 0.5x"),
        (720, 720, 0.5, "30d momentum 0.5x"),
    ]:
        try:
            be = run_baseline(oos, lookback=lb, hold=hd, frac=frac, cfg=cfg)
            _summary(label, be, pd.DataFrame({"pnl": [0.0]}))
        except Exception as e:
            log.info("  %s: skipped (%s)", label, e)

    # Gate criteria.
    gates = {
        "net_return > 0": s["return"] > 0,
        "martin > 0": s["martin"] > 0,
        "trades >= 5": s["n_trades"] >= 5,
        "beats_buy_hold": s.get("beats_bh", s["return"] > bh_ret),
        "max_dd < 25%": abs(s["max_dd"]) < 0.25,
    }
    if args.permutations > 0:
        p_val = permutation_p(
            oos["close"].to_numpy(dtype=float), s["return"], cfg, args.permutations, args.block, args.seed
        )
        gates["block_permutation p < 0.05"] = p_val < 0.05
        log.info("block permutation p = %.4f (n=%d, block=%d)", p_val, args.permutations, args.block)

    log.info("=== GATE ===")
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
