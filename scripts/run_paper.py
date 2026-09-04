#!/usr/bin/env python3
"""Run paper verification: compare research (vectorised) vs live (stateful) signals.

Loads the frozen config from configs/passed.yaml, runs both signal paths for
each symbol, and reports signal agreement + combined portfolio metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.engine import EventBacktest  # noqa: E402
from live.strategy import FrozenMomentumStrategy  # noqa: E402
from research.config import PassedConfig  # noqa: E402
from research.costs import CostModel  # noqa: E402
from research.data import load_dataset  # noqa: E402
from research.log import get_logger  # noqa: E402
from research.metrics import martin_ratio, max_drawdown  # noqa: E402
from research.momentum import constant_sizes, ts_momentum_pip_exit  # noqa: E402

log = get_logger("paper")


def main() -> None:
    cfg = PassedConfig.load(ROOT / "configs" / "passed.yaml")
    symbols = cfg.symbols
    train_n = 17520  # from calibration
    frac_total = cfg.notional_frac
    n_sym = len(symbols)
    frac_each = frac_total / n_sym

    cost = CostModel(
        taker_fee=cfg.costs.taker_fee,
        slippage_bps=cfg.costs.slippage_bps,
        use_funding=cfg.costs.use_funding,
    )

    total_capital = cfg.backtest.start_equity
    sub_capital = total_capital / n_sym

    headers = (
        f"{'symbol':<10} {'bars':<8} {'agree_f':<10} {'diff':<8} "
        f"{'research_ret':<14} {'live_ret':<14}"
    )
    log.info(headers)
    log.info("-" * len(headers))

    sig_agree_all = True
    research_equities = []
    live_equities = []

    for sym in symbols:
        kp = ROOT / f"data/{sym}_1h.csv"
        fp = ROOT / f"data/{sym}_funding.csv"
        if not kp.exists() or not fp.exists():
            log.warning("  %s 数据缺失跳过", sym)
            continue

        df = load_dataset(kp, fp)
        oos = df.iloc[train_n:].reset_index(drop=True)
        close = oos["close"].to_numpy(dtype=float)
        bars = len(close)

        sig_research = ts_momentum_pip_exit(
            close,
            lookback=cfg.lookback,
            hold=cfg.hold,
            mode=cfg.mode,
            pip_n=cfg.pip_n,
            min_bars=cfg.min_bars,
            fit_exit=cfg.fit_exit,
            use_pip=True,
            stop_pct=cfg.stop_pct,
            trail_arm=cfg.trail_arm,
            trail_giveback=cfg.trail_giveback,
        )

        strat = FrozenMomentumStrategy(
            lookback=cfg.lookback,
            hold=cfg.hold,
            mode=cfg.mode,
            exit_mode=cfg.exit_mode,
            pip_n=cfg.pip_n,
            min_bars=cfg.min_bars,
            fit_exit=cfg.fit_exit,
            stop_pct=cfg.stop_pct,
            trail_arm=cfg.trail_arm,
            trail_giveback=cfg.trail_giveback,
        )
        n = len(close)
        sig_live = np.zeros(n)
        for i in range(n):
            sig_live[i] = strat.on_bar_close(close[: i + 1])

        agree = float(np.mean(sig_live == sig_research))
        diff = int(np.sum(sig_live != sig_research))
        if agree < 1.0:
            sig_agree_all = False

        bt = EventBacktest(
            cost=cost,
            start_equity=sub_capital,
            notional_frac=frac_each,
            skip_after_losing_fold=False,
        )
        sizes = constant_sizes(bars, frac_each)
        r_r = bt.run_signals(oos, sig_research, sizes)
        r_l = bt.run_signals(oos, sig_live, sizes)

        research_equities.append(r_r.equity)
        live_equities.append(r_l.equity)

        log.info(
            "  %-10s %-8d %-10.4f %-8d %-14.6f %-14.6f",
            sym, bars, agree, diff, r_r.summary["return"], r_l.summary["return"],
        )

    # Combined portfolio equity. Symbols have different OOS lengths
    # (BTC 40632 / ETH 40824 / SOL 34529 bars), so a raw np.sum along axis=0
    # raises "inhomogeneous shape". Truncate to the shortest length and sum.
    combined_r = np.array([])
    combined_l = np.array([])
    if research_equities and live_equities:
        min_len = min(len(e) for e in research_equities + live_equities)
        combined_r = np.sum([e[:min_len] for e in research_equities], axis=0)
        combined_l = np.sum([e[:min_len] for e in live_equities], axis=0)
    log.info("")
    combined_ret_l = float(combined_l[-1] / total_capital - 1.0) if len(combined_l) else 0.0
    combined_dd_l = max_drawdown(combined_l) if len(combined_l) else 0.0
    log.info("组合 (3 symbolic, 等权):")
    log.info("  return:      %.6f", combined_ret_l)
    log.info("  max_dd:      %.6f", combined_dd_l)
    combined_martin_l = martin_ratio(np.log(np.maximum(combined_l, 1e-12))) if len(combined_l) else 0.0
    log.info("  martin:      %.4f", combined_martin_l)
    eq_ok = abs(combined_r[-1] - combined_l[-1]) < 1e-3
    log.info("  research ≡ live: %s", "✓" if eq_ok else "✗")

    # Compare against passed.json (BTC-only reference)
    frozen = json.loads((ROOT / "models/passed.json").read_text())
    log.info("")
    log.info("对比 passed.json（单币种 BTC-only，仅供参考）:")
    log.info("  oos_return:   expected=%.6f  actual=%.6f", frozen["oos_return"], combined_ret_l)
    log.info("  oos_max_dd:   expected=%.6f  actual=%.6f", frozen["oos_max_dd"], combined_dd_l)
    log.info("  oos_martin:   expected=%.4f  actual=%.4f", frozen["oos_martin"], combined_martin_l)
    log.info("  (差异来源：3 币种组合 ≠ 1 币种，pass 预期不同)")

    # Save summary
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    summary = {
        "symbols": symbols,
        "signal_agree": sig_agree_all,
        "combined_return": combined_ret_l,
        "combined_max_dd": combined_dd_l,
        "combined_martin": combined_martin_l,
    }
    (report_dir / "paper_summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    log.info("结果已保存到 %s/paper_summary.json", report_dir)

    # The research (vectorized) and live (stateful) signal paths MUST agree
    # bar-for-bar. If they don't, the gate metrics are computed on a signal
    # the live bot never trades — fail loudly so CI catches regressions.
    if not sig_agree_all:
        log.error("信号一致率 < 100%%，回测/实盘语义漂移，拒绝接受")
        sys.exit(1)


if __name__ == "__main__":
    main()
