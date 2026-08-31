"""Config validation for crypto-pip-trader.

Provides typed dataclass wrappers around YAML config files so callers
don't have to deal with raw dicts.  Raises early on missing/wrong types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CostConfig:
    taker_fee: float = 0.0004
    slippage_bps: float = 1.0
    use_funding: bool = True


@dataclass
class RiskConfig:
    notional_frac: float = 0.5
    max_leverage: float = 2.0
    daily_loss_halt: float = 0.05
    max_drawdown_halt: float = 0.15


@dataclass
class BacktestConfig:
    start_equity: float = 1000.0
    fill: str = "next_open"


def _f(v: Any, default: float, cls: type = float) -> float:
    if v is None:
        return default
    return cls(v)


def _i(v: Any, default: int) -> int:
    if v is None:
        return default
    return int(v)


def _s(v: Any, default: str) -> str:
    if v is None:
        return default
    return str(v)


@dataclass
class PassedConfig:
    """Typed wrapper around configs/passed.yaml."""

    symbol: str = "BTCUSDT"
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    interval: str = "1h"
    market: str = "usdm"
    strategy: str = ""
    lookback: int = 720
    hold: int = 720
    mode: str = "long_short"
    exit_mode: str = "pip"
    pip_n: int = 5
    min_bars: int = 24
    fit_exit: float = 0.0
    stop_pct: float | None = None
    trail_arm: float | None = None
    trail_giveback: float | None = None
    use_exchange_stop: bool = True
    notional_frac: float = 0.25
    costs: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @classmethod
    def load(cls, path: str | Path) -> PassedConfig:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"not a valid YAML map: {path}")
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> PassedConfig:
        costs = CostConfig(
            taker_fee=_f(d.get("costs", {}).get("taker_fee"), 0.0004),
            slippage_bps=_f(d.get("costs", {}).get("slippage_bps"), 1.0),
            use_funding=bool(d.get("costs", {}).get("use_funding", True)),
        )
        risk = RiskConfig(
            notional_frac=_f(d.get("risk", {}).get("notional_frac"), 0.5),
            max_leverage=_f(d.get("risk", {}).get("max_leverage"), 2.0),
            daily_loss_halt=_f(d.get("risk", {}).get("daily_loss_halt"), 0.05),
            max_drawdown_halt=_f(d.get("risk", {}).get("max_drawdown_halt"), 0.15),
        )
        backtest = BacktestConfig(
            start_equity=_f(d.get("backtest", {}).get("start_equity"), 1000.0),
            fill=_s(d.get("backtest", {}).get("fill"), "next_open"),
        )

        stop_pct = d.get("stop_pct")
        trail_arm = d.get("trail_arm")
        trail_giveback = d.get("trail_giveback")

        return cls(
            symbol=_s(d.get("symbol"), "BTCUSDT"),
            symbols=list(d.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])),
            interval=_s(d.get("interval"), "1h"),
            market=_s(d.get("market"), "usdm"),
            strategy=_s(d.get("strategy"), ""),
            lookback=_i(d.get("lookback"), 720),
            hold=_i(d.get("hold"), 720),
            mode=_s(d.get("mode"), "long_short"),
            exit_mode=_s(d.get("exit_mode"), "pip"),
            pip_n=_i(d.get("pip_n"), 5),
            min_bars=_i(d.get("min_bars"), 24),
            fit_exit=_f(d.get("fit_exit"), 0.0),
            stop_pct=_f(stop_pct, 0.0) if stop_pct is not None else None,
            trail_arm=_f(trail_arm, 0.0) if trail_arm is not None else None,
            trail_giveback=_f(trail_giveback, 0.0) if trail_giveback is not None else None,
            use_exchange_stop=bool(d.get("use_exchange_stop", True)),
            notional_frac=_f(d.get("notional_frac"), 0.25),
            costs=costs,
            risk=risk,
            backtest=backtest,
        )
