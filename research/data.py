from __future__ import annotations

import io
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

VISION = "https://data.binance.vision/data/futures/um"
PUBLIC_SPOT = "https://data-api.binance.vision"
KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def _get_bytes(url: str, retries: int = 6) -> bytes | None:
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.content
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    if last_err:
        raise last_err
    return None


def _read_zip_csv(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            sample = fh.read(256)
            fh.seek(0)
            header = None
            text = sample.decode("utf-8", errors="ignore")
            if "open_time" in text or "fundingTime" in text or "funding_rate" in text:
                header = 0
            return pd.read_csv(fh, header=header)


def _month_range(start: date, end: date) -> list[tuple[int, int]]:
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def _day_range(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur = date.fromordinal(cur.toordinal() + 1)
    return days


def _cache_dir(out_path: Path, kind: str) -> Path:
    d = out_path.parent / "_cache" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _normalize_klines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "open_time" not in df.columns:
        n = min(len(df.columns), len(KLINE_COLS))
        df = df.iloc[:, :n]
        df.columns = KLINE_COLS[:n]
    keep = [c for c in KLINE_COLS if c in df.columns]
    df = df[keep]
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
        ot = pd.to_numeric(df["open_time"], errors="coerce")
        if ot.notna().mean() > 0.9:
            unit = "ms" if float(ot.max()) > 1e12 else "s"
            df["open_time"] = pd.to_datetime(ot, unit=unit, utc=True)
        else:
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    elif df["open_time"].dt.tz is None:
        df["open_time"] = df["open_time"].dt.tz_localize("UTC")
    if "close_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["close_time"]):
        ct = pd.to_numeric(df["close_time"], errors="coerce")
        if ct.notna().mean() > 0.9:
            unit = "ms" if float(ct.max()) > 1e12 else "s"
            df["close_time"] = pd.to_datetime(ct, unit=unit, utc=True)
        else:
            df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["open_time", "close"]).drop_duplicates(subset=["open_time"]).sort_values("open_time")
    return df


def _normalize_funding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("fundingtime", "calc_time", "time"):
            rename[c] = "fundingTime"
        if cl in ("fundingrate", "funding_rate", "last_funding_rate"):
            rename[c] = "fundingRate"
    df = df.rename(columns=rename)
    if "fundingTime" not in df.columns:
        if df.shape[1] >= 3:
            df.columns = ["symbol", "fundingTime", "fundingRate"] + list(df.columns[3:])
        elif df.shape[1] == 2:
            df.columns = ["fundingTime", "fundingRate"]
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["fundingTime"]):
        ft = pd.to_numeric(df["fundingTime"], errors="coerce")
        if ft.notna().mean() > 0.9:
            unit = "ms" if float(ft.max()) > 1e12 else "s"
            df["fundingTime"] = pd.to_datetime(ft, unit=unit, utc=True)
        else:
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], utc=True, errors="coerce")
    elif df["fundingTime"].dt.tz is None:
        df["fundingTime"] = df["fundingTime"].dt.tz_localize("UTC")
    return (
        df.dropna(subset=["fundingTime", "fundingRate"])
        .drop_duplicates(subset=["fundingTime"])
        .sort_values("fundingTime")
    )


def _fetch_cached(url: str, cache_path: Path) -> pd.DataFrame | None:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    raw = _get_bytes(url)
    if raw is None:
        return None
    df = _read_zip_csv(raw)
    df.to_csv(cache_path, index=False)
    return df


def download_klines(
    symbol: str,
    interval: str,
    start: str,
    out_path: str | Path,
    end: str | None = None,
) -> pd.DataFrame:
    start_d = pd.Timestamp(start, tz="UTC").date()
    end_d = pd.Timestamp(end, tz="UTC").date() if end else pd.Timestamp.now(tz="UTC").date()
    out = Path(out_path)
    cache = _cache_dir(out, "klines")
    frames = []

    months = _month_range(start_d.replace(day=1), date(end_d.year, end_d.month, 1))
    complete_months = months[:-1] if months else []
    for y, m in complete_months:
        ym = f"{y:04d}-{m:02d}"
        url = f"{VISION}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
        path = cache / f"{symbol}-{interval}-{ym}.csv"
        df = _fetch_cached(url, path)
        if df is None:
            print(f"skip missing monthly kline {ym}")
            continue
        frames.append(df)
        print(f"kline monthly {ym}")
        time.sleep(0.03)

    if months:
        y, m = months[-1]
        for d in _day_range(date(y, m, 1), end_d):
            ds = d.isoformat()
            url = f"{VISION}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{ds}.zip"
            path = cache / f"{symbol}-{interval}-{ds}.csv"
            df = _fetch_cached(url, path)
            if df is None:
                continue
            frames.append(df)
            print(f"kline daily {ds}")
            time.sleep(0.02)

    if not frames:
        raise RuntimeError("no klines downloaded")
    df = _normalize_klines(pd.concat([_normalize_klines(x) for x in frames], ignore_index=True))
    start_ts = pd.Timestamp(start, tz="UTC")
    df = df[df["open_time"] >= start_ts]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def download_funding(
    symbol: str,
    start: str,
    out_path: str | Path,
    end: str | None = None,
) -> pd.DataFrame:
    start_d = pd.Timestamp(start, tz="UTC").date()
    end_d = pd.Timestamp(end, tz="UTC").date() if end else pd.Timestamp.now(tz="UTC").date()
    out = Path(out_path)
    cache = _cache_dir(out, "funding")
    frames = []
    months = _month_range(start_d.replace(day=1), date(end_d.year, end_d.month, 1))
    complete_months = months[:-1] if months else []
    for y, m in complete_months:
        ym = f"{y:04d}-{m:02d}"
        url = f"{VISION}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
        path = cache / f"{symbol}-fundingRate-{ym}.csv"
        df = _fetch_cached(url, path)
        if df is None:
            print(f"skip missing monthly funding {ym}")
            continue
        frames.append(df)
        print(f"funding monthly {ym}")
        time.sleep(0.03)

    if months:
        y, m = months[-1]
        for d in _day_range(date(y, m, 1), end_d):
            ds = d.isoformat()
            url = f"{VISION}/daily/fundingRate/{symbol}/{symbol}-fundingRate-{ds}.zip"
            path = cache / f"{symbol}-fundingRate-{ds}.csv"
            df = _fetch_cached(url, path)
            if df is None:
                continue
            frames.append(df)
            print(f"funding daily {ds}")
            time.sleep(0.02)

    if not frames:
        raise RuntimeError("no funding downloaded")
    df = _normalize_funding(pd.concat([_normalize_funding(x) for x in frames], ignore_index=True))
    start_ts = pd.Timestamp(start, tz="UTC")
    df = df[df["fundingTime"] >= start_ts]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def fetch_public_spot_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 1000,
    closed_only: bool = True,
) -> pd.DataFrame:
    """Public spot klines. No API key. Futures REST is geo-blocked (HTTP 451)."""
    rows = []
    end_ms = None
    remain = max(1, int(limit))
    while remain > 0:
        batch_n = min(1000, remain)
        params = {"symbol": symbol, "interval": interval, "limit": batch_n}
        if end_ms is not None:
            params["endTime"] = end_ms
        r = requests.get(PUBLIC_SPOT + "/api/v3/klines", params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows = batch + rows
        first_open = int(batch[0][0])
        end_ms = first_open - 1
        remain -= len(batch)
        if len(batch) < batch_n:
            break
        time.sleep(0.05)
    if not rows:
        raise RuntimeError("no public klines")
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    df = _normalize_klines(df)
    if closed_only and "close_time" in df.columns:
        now = pd.Timestamp.now(tz="UTC")
        df = df[df["close_time"] < now]
    df["log_close"] = np.log(df["close"].to_numpy(dtype=float))
    df["funding_rate"] = 0.0
    df["source"] = "spot_public"
    return df.reset_index(drop=True)


def load_dataset(kline_path: str | Path, funding_path: str | Path) -> pd.DataFrame:
    k = pd.read_csv(kline_path)
    k["open_time"] = pd.to_datetime(k["open_time"], utc=True, errors="coerce")
    if "close_time" in k.columns:
        k["close_time"] = pd.to_datetime(k["close_time"], utc=True, errors="coerce")
    f = pd.read_csv(funding_path)
    f["fundingTime"] = pd.to_datetime(f["fundingTime"], utc=True, errors="coerce")
    k = k.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    k["log_close"] = np.log(k["close"].to_numpy(dtype=float))
    k["funding_rate"] = 0.0
    if not f.empty:
        f = f.sort_values("fundingTime").drop_duplicates("fundingTime")
        merged = k.merge(
            f[["fundingTime", "fundingRate"]],
            left_on="open_time",
            right_on="fundingTime",
            how="left",
        )
        k["funding_rate"] = merged["fundingRate"].fillna(0.0).to_numpy()
    return k
