from __future__ import annotations

import numpy as np
import pandas as pd


def martin_ratio(log_rets: np.ndarray) -> float:
    rets = np.asarray(log_rets, dtype=float)
    rets = np.nan_to_num(rets, nan=0.0)
    rsum = float(np.sum(rets))
    short = False
    if rsum < 0.0:
        rets = rets * -1
        rsum *= -1
        short = True
    if len(rets) == 0 or rsum == 0.0:
        return 0.0
    csum = np.cumsum(rets)
    # Normalise the cumulative log-return to start at 0 so exp() can't overflow
    # on long series with large total return (e.g. 3-coin portfolio equity).
    # Martin is scale-invariant, so shifting csum by a constant is safe.
    csum = csum - csum[0]
    eq = np.exp(csum)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    sumsq = float(np.sum(dd ** 2.0))
    ulcer = (sumsq / len(rets)) ** 0.5
    martin = (0.0 if rsum == 0.0 else np.sign(rsum) * 1000000.0) if ulcer == 0.0 else rsum / ulcer
    if short:
        martin = -martin
    return float(martin)


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def summarize_trades(trade_log: pd.DataFrame, equity: np.ndarray) -> dict:
    n = int(len(trade_log)) if trade_log is not None else 0
    net = float(trade_log["pnl"].sum()) if n else 0.0
    wins = int((trade_log["pnl"] > 0).sum()) if n else 0
    return {
        "n_trades": n,
        "net_pnl": net,
        "win_rate": (wins / n) if n else 0.0,
        "avg_pnl": (net / n) if n else 0.0,
        "max_dd": max_drawdown(equity) if equity is not None and len(equity) else 0.0,
        "end_equity": float(equity[-1]) if equity is not None and len(equity) else 0.0,
    }
