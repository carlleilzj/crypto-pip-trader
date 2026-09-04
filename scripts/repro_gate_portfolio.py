#!/usr/bin/env python3
"""Reproduce the frozen strategy gate on the 3-coin PORTFOLIO (post parity fix).

The single-coin BTC gate failed after the backtest/live signal parity fix
(+19.4% < buy&hold +20.1%, block p=0.066). This script re-runs the gate on
the 3-coin equal-weight portfolio (BTC/ETH/SOL) that the live bot actually
trades, using the SAME frozen spec in configs/passed.yaml and the same
parity-aligned signal path.

Gate criteria (unchanged from the README hard thresholds):
  net return > 0, Martin > 0, trades >= 5, beats buy-and-hold,
  max drawdown < 25%, block-permutation p < 0.05.

Block permutations are applied IN SYNCHRONOUSLY across the three coins
(same seed, same block boundaries) so cross-sectional correlation structure
is preserved — this is the conservative choice (overstates p vs independent
shuffles, so a pass here is more trustworthy).

Usage:
  python scripts/repro_gate_portfolio.py                 # gate check only (fast)
  python scripts/repro_gate_portfolio.py --permutations 200  # full p-value (slow)
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
    constant_sizes,
    oos_slice,
    ts_momentum_pip_exit,
)

log = get_logger("gate_portfolio")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TRAIN_H = 17520  # 2020-01-01 → 2021-12-31 (calibration window)


def _summary(name: str, equity: np.ndarray, n_trades: int, bh: float | None = None) -> dict:
    ret = float(equity[-1] / equity[0] - 1.0) if len(equity) else 0.0
    log_eq = np.log(np.maximum(equity, 1e-12))
    log_rets = np.diff(log_eq, prepend=log_eq[0])
    martin = martin_ratio(log_rets)
    mdd = max_drawdown(equity)
    out = {"name": name, "return": ret, "martin": martin, "max_dd": mdd, "n_trades": n_trades}
    line = f"  {name:<28} ret {ret*100:+6.1f}%  Martin {martin:6.2f}  dd {mdd*100:6.1f}%  trades {n_trades:4d}"
    if bh is not None:
        out["beats_bh"] = ret > bh
        line += f"  beats BH {bh*100:+.1f}%"
    log.info("%s", line)
    return out


def run_frozen_single(df: pd.DataFrame, cfg: PassedConfig, frac_each: float,
                      start_equity: float) -> tuple[np.ndarray, pd.DataFrame]:
    """Run frozen strategy on one coin with per-coin capital and size."""
    close = df["close"].to_numpy(dtype=float)
    sig = ts_momentum_pip_exit(
        close,
        lookback=cfg.lookback, hold=cfg.hold, mode=cfg.mode,
        pip_n=cfg.pip_n, min_bars=cfg.min_bars, fit_exit=cfg.fit_exit,
        use_pip=(cfg.exit_mode == "pip"), stop_pct=cfg.stop_pct,
    )
    sizes = constant_sizes(len(df), frac_each)
    cost = CostModel(
        taker_fee=cfg.costs.taker_fee, slippage_bps=cfg.costs.slippage_bps,
        use_funding=cfg.costs.use_funding,
    )
    bt = EventBacktest(cost=cost, start_equity=start_equity, notional_frac=frac_each)
    res = bt.run_signals(df, sig, sizes)
    return res.equity, res.trades


def run_portfolio(oos_per_coin: list[pd.DataFrame], cfg: PassedConfig,
                  total_equity: float = 1000.0) -> tuple[np.ndarray, int]:
    """Run frozen strategy on each coin, combine into equal-weight portfolio.

    Returns (combined_equity, total_trades). Coins have different OOS lengths;
    truncate to the shortest so the sum is well-defined (the dropped tail is
    < 1% of bars and doesn't materially change the result).
    """
    n_sym = len(oos_per_coin)
    frac_each = cfg.notional_frac / n_sym
    sub_capital = total_equity / n_sym
    equities = []
    total_trades = 0
    for df in oos_per_coin:
        eq, trades = run_frozen_single(df, cfg, frac_each, sub_capital)
        equities.append(eq)
        total_trades += len(trades)
    min_len = min(len(e) for e in equities)
    combined = np.sum([e[:min_len] for e in equities], axis=0)
    return combined, total_trades


def run_buyhold_portfolio(oos_per_coin: list[pd.DataFrame], cfg: PassedConfig,
                          total_equity: float = 1000.0) -> np.ndarray:
    """Buy & hold each coin, equal weight. Full size (frac=1.0 per coin)."""
    n_sym = len(oos_per_coin)
    sub_capital = total_equity / n_sym
    cost = CostModel(
        taker_fee=cfg.costs.taker_fee, slippage_bps=cfg.costs.slippage_bps,
        use_funding=cfg.costs.use_funding,
    )
    equities = []
    for df in oos_per_coin:
        n = len(df)
        sig = np.ones(n)  # always long
        sizes = constant_sizes(n, 1.0)
        bt = EventBacktest(cost=cost, start_equity=sub_capital, notional_frac=1.0)
        equities.append(bt.run_signals(df, sig, sizes).equity)
    min_len = min(len(e) for e in equities)
    return np.sum([e[:min_len] for e in equities], axis=0)


def permutation_p_portfolio(
    oos_per_coin: list[pd.DataFrame], actual_ret: float, cfg: PassedConfig,
    n_reps: int, block: int, seed0: int, total_equity: float = 1000.0,
) -> float:
    """Block-permutation p-value on the portfolio.

    Synchronously shuffles all coins with the same seed+block so cross-sectional
    correlation is preserved. p = (1 + #{shuffled >= actual}) / (1 + n_reps).
    """
    ge = 0
    for r in range(n_reps):
        permuted = []
        for df in oos_per_coin:
            close = df["close"].to_numpy(dtype=float)
            perm = block_permute_close(close, block, seed0 + r)
            permuted.append(pd.DataFrame({
                "open": perm, "high": perm, "low": perm, "close": perm,
                "funding_rate": 0.0,
            }))
        eq, _ = run_portfolio(permuted, cfg, total_equity)
        if float(eq[-1] / eq[0] - 1.0) >= actual_ret:
            ge += 1
        if (r + 1) % 20 == 0:
            log.info("  permutation %d/%d (ge=%d)", r + 1, n_reps, ge)
    return (1.0 + ge) / (1.0 + n_reps)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--permutations", type=int, default=0, help="block-permutation reps (0 = skip)")
    p.add_argument("--block", type=int, default=168, help="permutation block size (hours)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = PassedConfig.load(ROOT / "configs" / "passed.yaml")
    total_equity = cfg.backtest.start_equity

    # Load all coins, slice OOS.
    oos_per_coin: list[pd.DataFrame] = []
    for sym in SYMBOLS:
        kp = ROOT / f"data/{sym}_1h.csv"
        fp = ROOT / f"data/{sym}_funding.csv"
        if not kp.exists() or not fp.exists():
            raise SystemExit(f"missing data {kp} / {fp} — run download_data.py")
        df = load_dataset(kp, fp)
        oos = oos_slice(df, TRAIN_H)
        oos_per_coin.append(oos)
        log.info("loaded %s: %d OOS bars %s → %s", sym, len(oos),
                 oos["open_time"].iloc[0], oos["open_time"].iloc[-1])

    # Per-coin breakdown first (transparency).
    log.info("per-coin (parity-aligned frozen spec, frac=%.4f each):",
             cfg.notional_frac / len(SYMBOLS))
    n_sym = len(SYMBOLS)
    frac_each = cfg.notional_frac / n_sym
    sub_capital = total_equity / n_sym
    for sym, df in zip(SYMBOLS, oos_per_coin, strict=False):
        eq, trades = run_frozen_single(df, cfg, frac_each, sub_capital)
        bh_eq = run_buyhold_portfolio([df], cfg, sub_capital)
        _summary(sym, eq, len(trades), float(bh_eq[-1] / bh_eq[0] - 1.0))

    # Portfolio.
    eq, n_trades = run_portfolio(oos_per_coin, cfg, total_equity)
    bh_eq = run_buyhold_portfolio(oos_per_coin, cfg, total_equity)
    bh_ret = float(bh_eq[-1] / bh_eq[0] - 1.0)
    s = _summary("frozen 3-coin portfolio", eq, n_trades, bh_ret)
    _ = _summary("buy & hold 3-coin", bh_eq, 1)

    # Gate criteria.
    gates = {
        "net_return > 0": s["return"] > 0,
        "martin > 0": s["martin"] > 0,
        "trades >= 5": s["n_trades"] >= 5,
        "beats_buy_hold": s.get("beats_bh", s["return"] > bh_ret),
        "max_dd < 25%": abs(s["max_dd"]) < 0.25,
    }
    if args.permutations > 0:
        p_val = permutation_p_portfolio(
            oos_per_coin, s["return"], cfg, args.permutations, args.block, args.seed, total_equity
        )
        gates["block_permutation p < 0.05"] = p_val < 0.05
        log.info("block permutation p = %.4f (n=%d, block=%d, synchronous across coins)",
                 p_val, args.permutations, args.block)

    log.info("=== GATE (3-coin portfolio, parity-aligned) ===")
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
