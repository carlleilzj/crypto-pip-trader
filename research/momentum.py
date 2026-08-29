from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 365.0 * 24.0


def _zscore(ys: np.ndarray) -> np.ndarray:
    a = np.asarray(ys, dtype=float)
    std = float(a.std())
    if std < 1e-12:
        return np.zeros_like(a)
    return (a - a.mean()) / std


def shape_fit(pred_y: np.ndarray, actual_y: np.ndarray) -> float:
    n = min(len(pred_y), len(actual_y))
    if n < 3:
        return 1.0
    a = _zscore(pred_y[:n])
    b = _zscore(actual_y[:n])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 1.0
    return float(np.dot(a, b) / denom)


def ts_momentum_signals(
    close: np.ndarray,
    lookback: int = 168,
    hold: int = 168,
    mode: str = "long_short",
) -> np.ndarray:
    """Pre-specified time-series momentum. Sign of past return, held `hold` bars.

    mode='long_short': +1 / -1
    mode='long_flat': +1 / 0  (absolute momentum; cash when past return < 0)
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    sig = np.zeros(n)
    i = lookback
    while i < n:
        past = c[i] / c[i - lookback] - 1.0
        if past > 0:
            side = 1.0
        elif mode == "long_flat":
            side = 0.0
        else:
            side = -1.0
        end = min(i + hold, n)
        sig[i:end] = side
        i = end
    return sig


def ts_momentum_pip_exit(
    close: np.ndarray,
    lookback: int = 720,
    hold: int = 720,
    mode: str = "long_short",
    pip_n: int = 5,
    min_bars: int = 24,
    fit_exit: float = 0.0,
    use_pip: bool = True,
    stop_pct: float | None = None,
    trail_arm: float | None = None,
    trail_giveback: float | None = None,
) -> np.ndarray:
    """Same entry as ts momentum; flatten on PIP divergence and/or price stops.

    fit_exit=0 means cosine(z(pred), z(actual)) < 0 (shape reversed).
    stop_pct: adverse move from entry, e.g. 0.08.
    trail_arm / trail_giveback: after favorable move >= trail_arm, flatten if
    price gives back trail_giveback from the in-trade extreme.
    """
    from research.pips import find_pips

    c = np.asarray(close, dtype=float)
    n = len(c)
    sig = np.zeros(n)
    i = lookback
    while i < n:
        past = c[i] / c[i - lookback] - 1.0
        if past > 0:
            side = 1.0
        elif mode == "long_flat":
            side = 0.0
        else:
            side = -1.0
        if side == 0.0:
            i += 1
            continue
        pred_w = c[i - min(24, lookback) + 1 : i + 1]
        pred_y: list[float] | np.ndarray
        if len(pred_w) >= pip_n:
            _, pred_y = find_pips(pred_w, pip_n, 3)
        else:
            pred_y = pred_w
        end = min(i + hold, n)
        exit_i = end
        entry = float(c[i])
        extreme = entry
        for j in range(i + 1, end):
            px = float(c[j])
            if side > 0:
                extreme = max(extreme, px)
                pnl = px / entry - 1.0
                dd = 1.0 - px / extreme if extreme > 0 else 0.0
            else:
                extreme = min(extreme, px)
                pnl = 1.0 - px / entry
                dd = px / extreme - 1.0 if extreme > 0 else 0.0
            if stop_pct is not None and pnl <= -stop_pct:
                exit_i = j
                break
            if trail_arm is not None and trail_giveback is not None and pnl >= trail_arm and dd >= trail_giveback:
                exit_i = j
                break
            if use_pip and (j - i) >= min_bars and (j - i - min_bars) % 12 == 0:
                actual_w = c[i : j + 1]
                k = min(pip_n, len(actual_w))
                if k >= 3:
                    _, actual_y = find_pips(actual_w, k, 3)
                    if shape_fit(np.asarray(pred_y), np.asarray(actual_y)) < fit_exit:
                        exit_i = j
                        break
        sig[i:exit_i] = side
        i = exit_i if exit_i > i else i + 1
    return sig


def constant_sizes(n: int, frac: float) -> np.ndarray:
    return np.full(n, float(frac), dtype=float)


def vol_target_sizes(
    close: np.ndarray,
    target_vol: float = 0.40,
    vol_lookback: int = 168,
    min_size: float = 0.10,
    max_size: float = 1.00,
    signals: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse-vol sizing. If `signals` is given, size is frozen until the signal changes."""
    c = np.asarray(close, dtype=float)
    n = len(c)
    logret = np.zeros(n)
    logret[1:] = np.diff(np.log(np.maximum(c, 1e-12)))
    raw = np.full(n, min_size, dtype=float)
    for i in range(vol_lookback, n):
        rv = float(np.std(logret[i - vol_lookback + 1 : i + 1], ddof=1))
        ann = rv * np.sqrt(HOURS_PER_YEAR)
        if ann <= 1e-8:
            raw[i] = max_size
        else:
            raw[i] = float(np.clip(target_vol / ann, min_size, max_size))
    raw[:vol_lookback] = raw[vol_lookback] if n > vol_lookback else min_size
    if signals is None:
        return raw
    sig = np.asarray(signals, dtype=float)
    sizes = np.zeros(n)
    last = 0.0
    held = min_size
    for i in range(n):
        if sig[i] != last:
            held = raw[i]
            last = sig[i]
        sizes[i] = held if sig[i] != 0 else 0.0
    return sizes


def block_permute_close(close: np.ndarray, block: int, seed: int) -> np.ndarray:
    """Shuffle log-return blocks to keep short-horizon autocorrelation."""
    diffs = np.diff(np.log(np.asarray(close, dtype=float)))
    n = len(diffs)
    if block <= 1 or block >= n:
        rng = np.random.default_rng(seed)
        rng.shuffle(diffs)
    else:
        blocks = [diffs[i : i + block] for i in range(0, n, block)]
        rng = np.random.default_rng(seed)
        rng.shuffle(blocks)
        diffs = np.concatenate(blocks)[:n]
    logp = np.concatenate([[np.log(close[0])], diffs]).cumsum()
    return np.exp(logp)


def buy_hold_signals(n: int) -> np.ndarray:
    return np.ones(n, dtype=float)


def calibrate_notional(
    train_dd: float,
    base_frac: float,
    target_dd: float = 0.20,
    min_frac: float = 0.05,
) -> float:
    """Scale position from train-window drawdown only. Never looks at OOS."""
    dd = abs(float(train_dd))
    if dd < 1e-9:
        return float(base_frac)
    scaled = base_frac * min(1.0, target_dd / dd)
    return float(np.clip(scaled, min_frac, base_frac))


def oos_slice(df: pd.DataFrame, train_hours: int) -> pd.DataFrame:
    if train_hours >= len(df):
        return df.iloc[0:0].copy()
    return df.iloc[train_hours:].reset_index(drop=True)
