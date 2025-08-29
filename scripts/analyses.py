#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses pipeline – scanner for pre-breakouts and breakouts (USDC universe)

- Universe: all Binance Spot pairs with quoteAsset == USDC
- Excludes bases that are stablecoins and leveraged/multiplier tokens
- BTC reference: BTCUSDC
- Emits public_runs/latest/summary.json and timestamped copy
- 'missing_binance' is always an empty list to keep the contract stable
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# ----------------------------- CONFIG ---------------------------------

TZ_NAME = "Europe/Prague"
TZ = ZoneInfo(TZ_NAME) if ZoneInfo else timezone.utc

# Binance public data endpoint (no auth)
BASE_URL = "https://data-api.binance.vision"

# Intervals & limits
INTERVALS = {
    "4h": ("4h", 500),
    "1h": ("1h", 600),
    "15m": ("15m", 400),
    "5m": ("5m", 400),
}

# ATR / EMA parameters
ATR_LEN = 14
EMA_FAST = 20
EMA_SLOW = 200
SWING_LOOKBACK = 20  # HH20 / LL20

# Pre-breakout heuristics (env-overridable)
PROX_ATR_MIN    = float(os.getenv("PROX_ATR_MIN",    "0.05"))
PROX_ATR_MAX    = float(os.getenv("PROX_ATR_MAX",    "0.35"))
VOL_Z_MIN_PRE   = float(os.getenv("VOL_Z_MIN_PRE",   "1.2"))
VOL_Z_MIN_BREAK = float(os.getenv("VOL_Z_MIN_BREAK", "1.5"))   # relaxed default
BREAK_BUFFER_ATR= float(os.getenv("BREAK_BUFFER_ATR","0.06"))  # relaxed default
RELAX_B_HIGH    = os.getenv("RELAX_B_HIGH","1") == "1"         # use 1h HIGH for confirmation


# Ranking
MAX_CANDIDATES = 10

# Filters
STABLES = {"USDT", "USDC", "DAI", "TUSD", "USDP", "BUSD", "FDUSD", "PYUSD"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S")

# -------------------------- HELPERS -----------------------------------

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[analyses {ts}] {msg}", flush=True)

def human_time(ts: Optional[datetime] = None) -> str:
    dt = ts or datetime.now(timezone.utc).astimezone(TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def safe_get(url: str, params: Optional[dict] = None, retries: int = 3, timeout: int = 20, sleep_s: float = 0.6) -> Optional[requests.Response]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (418, 429, 451, 403, 500, 504):
                time.sleep(sleep_s * attempt)
        except requests.RequestException:
            time.sleep(sleep_s * attempt)
    return None

def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series[:]
    alpha = 2.0 / (period + 1.0)
    out: List[float] = []
    prev = series[0]
    out.append(prev)
    for x in series[1:]:
        prev = alpha * x + (1 - alpha) * prev
        out.append(prev)
    return out

def atr(highs: List[float], lows: List[float], closes: List[float], period: int = ATR_LEN) -> List[float]:
    trs = [0.0]
    for i in range(1, len(highs)):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return ema(trs, period)

def swing_high(series: List[float], lookback: int) -> List[float]:
    return [max(series[max(0, i - lookback + 1): i + 1]) for i in range(len(series))]

def swing_low(series: List[float], lookback: int) -> List[float]:
    return [min(series[max(0, i - lookback + 1): i + 1]) for i in range(len(series))]

def pct_change(series: List[float], n: int) -> float:
    if len(series) < n + 1:
        return 0.0
    a, b = series[-n - 1], series[-1]
    if a == 0:
        return 0.0
    return (b - a) / a

def vol_zscore(vols: List[float], window: int = 50) -> float:
    if len(vols) < window + 1:
        return 0.0
    sample = vols[-(window + 1):-1]
    mu = statistics.fmean(sample)
    sd = statistics.pstdev(sample)
    if sd == 0:
        return 0.0
    return (vols[-1] - mu) / sd

def fetch_exchange_usdc_symbols() -> Dict[str, dict]:
    """Return dict of USDC spot symbols metadata keyed by symbol (e.g., 'LPTUSDC')."""
    url = f"{BASE_URL}/api/v3/exchangeInfo"
    r = safe_get(url, retries=3, timeout=25)
    if not r:
        return {}
    data = r.json()
    out: Dict[str, dict] = {}
    for s in data.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != "USDC":
            continue
        if not s.get("isSpotTradingAllowed", True):
            continue
        base = s.get("baseAsset", "").upper()
        if base in STABLES:
            continue
        if any(base.endswith(sfx) for sfx in LEVERAGED_SUFFIXES):
            continue
        out[s["symbol"]] = s
    return out

def fetch_klines(symbol: str, interval: str, limit: int) -> Optional[List[List[Any]]]:
    """Fetch raw klines array for a given symbol/interval."""
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = safe_get(url, params=params, retries=3, timeout=20)
    if not r:
        return None
    try:
        data = r.json()
        if isinstance(data, list):
            return data
    except Exception:
        return None
    return None

def parse_klines(raw: List[List[Any]]) -> Dict[str, List[float]]:
    """Return dict with open_time(ms), open, high, low, close, volume as lists."""
    return {
        "open_time": [int(row[0]) for row in raw],
        "open": [float(row[1]) for row in raw],
        "high": [float(row[2]) for row in raw],
        "low": [float(row[3]) for row in raw],
        "close": [float(row[4]) for row in raw],
        "volume": [float(row[5]) for row in raw],
    }

# -------------------------- CORE LOGIC --------------------------------

def analyze_symbol(symbol: str, btc_1h_close: List[float]) -> Optional[dict]:
    req: Dict[str, Dict[str, List[float]]] = {}
    for tf, (interval, limit) in INTERVALS.items():
        raw = fetch_klines(symbol, interval, limit)
        if not raw or len(raw) < max(EMA_SLOW + 5, SWING_LOOKBACK + 5):
            return None
        req[tf] = parse_klines(raw)

    one = req["1h"]
    four = req["4h"]
    fifteen = req["15m"]
    five = req["5m"]

    close_1h = one["close"]
    high_1h = one["high"]
    low_1h = one["low"]
    vol_1h = one["volume"]

    # Trend filters
    ema20_1h = ema(close_1h, EMA_FAST)
    ema200_1h = ema(close_1h, EMA_SLOW)
    ema20_4h = ema(four["close"], EMA_FAST)
    ema200_4h = ema(four["close"], EMA_SLOW)
    if len(ema200_1h) < 5 or len(ema200_4h) < 5:
        return None

    ema20_slope_1h = ema20_1h[-1] - ema20_1h[-4]
    ema20_slope_4h = ema20_4h[-1] - ema20_4h[-4]

    trend_ok = (
        ema20_slope_1h > 0
        and ema20_slope_4h > 0
        and close_1h[-1] > ema200_1h[-1]
        and four["close"][-1] > ema200_4h[-1]
    )

    # ATR & swings on 1h
    atr_1h = atr(high_1h, low_1h, close_1h, ATR_LEN)
    if atr_1h[-1] <= 0:
        return None

    hh20 = swing_high(high_1h, SWING_LOOKBACK)
    ll20 = swing_low(low_1h, SWING_LOOKBACK)

    # Volume context
    vz = vol_zscore(vol_1h, window=50)

    # Pre-breakout proximity (to HH20)
    prox = (hh20[-1] - close_1h[-1]) / max(1e-9, atr_1h[-1])  # in ATRs
    atr_rising = atr_1h[-1] > atr_1h[-2] > atr_1h[-3]

    # Momentum confirms (lower TFs): price > EMA20 and EMA20 sloping up
    def mom_ok(tf_data: Dict[str, List[float]]) -> bool:
        c = tf_data["close"]
        e20 = ema(c, EMA_FAST)
        if len(e20) < 5:
            return False
        return (c[-1] > e20[-1]) and (e20[-1] > e20[-3])

    mom15 = mom_ok(fifteen)
    mom5 = mom_ok(five)
    lower_tf_ok = mom15 and mom5

    # RS vs BTC (1h)
    rs_strength = 0.0
    if btc_1h_close and len(btc_1h_close) == len(close_1h):
        rs_series = [c / b if b != 0 else 0.0 for c, b in zip(close_1h, btc_1h_close)]
        rs_strength = pct_change(rs_series, n=10)

    last_close = close_1h[-1]
    last_low = low_1h[-1]
    last_atr = atr_1h[-1]

    reasons: List[str] = []
    if trend_ok:
        reasons.append("TrendOK(1h&4h)")
    if atr_rising:
        reasons.append("ATR up")
    if vz > 0:
        reasons.append(f"VolZ={vz:.2f}")
    if lower_tf_ok:
        reasons.append("LowerTF OK")

    score = 0.0
    if vz > 0:
        score += vz
    if prox > 0:
        score += 1.0 / (min(max(prox, 0.01), 2.0))
    score += max(0.0, 5.0 * rs_strength)
    if lower_tf_ok:
        score += 0.5

    breakout_level = hh20[-1]
    last_high = high_1h[-1]
    confirm_price = last_high if RELAX_B_HIGH else last_close
    confirmed_breakout = (
        trend_ok
        and confirm_price > (breakout_level + BREAK_BUFFER_ATR * last_atr)
        and vz >= VOL_Z_MIN_BREAK
        and lower_tf_ok
    )

    pre_breakout = (
        trend_ok
        and atr_rising
        and PROX_ATR_MIN <= prox <= PROX_ATR_MAX
        and vz >= VOL_Z_MIN_PRE
    )

    entry = breakout_level + BREAK_BUFFER_ATR * last_atr
    stop = last_low  # breakout bar low proxy on 1h
    t1 = entry + 0.8 * last_atr
    t2 = entry + 1.5 * last_atr

    out: Dict[str, Any] = {
        "symbol": symbol,
        "last": last_close,
        "atr": last_atr,
        "hh20": breakout_level,
        "ll20": ll20[-1],
        "entry": entry,
        "stop": stop,
        "t1": t1,
        "t2": t2,
        "vol_z": vz,
        "prox_atr": prox,
        "trend_ok": trend_ok,
        "lower_tf_ok": lower_tf_ok,
        "rs10": rs_strength,
        "score": score,
        "reasons": reasons,
    }

    if confirmed_breakout:
        out["signal"] = "B"
    elif pre_breakout:
        out["signal"] = "C"
    else:
        out["signal"] = "N"

    return out

def write_summary(payload: dict, dest_latest: Path) -> None:
    ensure_dirs(dest_latest)
    with dest_latest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

# -------------------------- MAIN --------------------------------------

def build_universe() -> Tuple[Dict[str, str], List[str]]:
    """
    Build mapping {base: symbol} for all Binance USDC spot pairs, excluding:
      - base in STABLES
      - leveraged tokens with suffixes in LEVERAGED_SUFFIXES
    Returns (mapping, []) where the missing list is always empty (contract compatibility).
    """
    exch = fetch_exchange_usdc_symbols()
    if not exch:
        return {}, []
    mapping: Dict[str, str] = {}
    for sym, meta in exch.items():
        base = meta.get("baseAsset", "").upper()
        if not base:
            continue
        mapping[base] = sym
    return mapping, []

def compute_regime(btc_1h: Dict[str, List[float]], btc_4h: Dict[str, List[float]]) -> Tuple[bool, str]:
    try:
        ema200_4h = ema(btc_4h["close"], EMA_SLOW)
        ema20_4h = ema(btc_4h["close"], EMA_FAST)
        ema200_1h = ema(btc_1h["close"], EMA_SLOW)
        ema20_1h = ema(btc_1h["close"], EMA_FAST)
        if len(ema200_4h) > 4 and len(ema200_1h) > 4:
            slope4 = ema20_4h[-1] - ema20_4h[-4]
            slope1 = ema20_1h[-1] - ema20_1h[-4]
            if (btc_4h["close"][-1] > ema200_4h[-1]) and (btc_1h["close"][-1] > ema200_1h[-1]) and slope4 > 0 and slope1 > 0:
                return True, "BTC uptrend (4h & 1h)"
            return False, "BTC not in uptrend"
        return False, "insufficient data"
    except Exception:
        return False, "regime calc error"

def run_pipeline(mode: str) -> dict:
    started = datetime.now(timezone.utc).astimezone(TZ)

    mapping, _ = build_universe()
    if not mapping:
        return {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "exchangeInfo unavailable or no eligible symbols"},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": 0, "eligible": 0, "skipped": {"no_data": [], "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode},
        }

    # RS baseline: BTCUSDC
    btc_symbol = "BTCUSDC"
    btc_raw_1h = fetch_klines(btc_symbol, INTERVALS["1h"][0], INTERVALS["1h"][1])
    btc_raw_4h = fetch_klines(btc_symbol, INTERVALS["4h"][0], INTERVALS["4h"][1])
    if not btc_raw_1h or not btc_raw_4h:
        return {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "BTC data unavailable"},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": 0, "eligible": len(mapping), "skipped": {"no_data": [], "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode},
        }

    btc_1h = parse_klines(btc_raw_1h)
    btc_4h = parse_klines(btc_raw_4h)
    regime_ok, regime_reason = compute_regime(btc_1h, btc_4h)
    btc_close_1h = btc_1h["close"]

    candidates: List[dict] = []
    confirmed: List[dict] = []
    no_data: List[str] = []
    scanned = 0

    items = list(mapping.items())  # [(base, symbol)]
    if mode == "light-fast":
        items = items[::2]  # simple thinning
    elif mode == "light-hourly":
        items = items[::3]

    for base, sym in items:
        scanned += 1
        info = analyze_symbol(sym, btc_close_1h)
        time.sleep(0.03)  # pacing
        if not info:
            no_data.append(base)
            continue
        info["ticker"] = base
        info["rotation_exempt"] = False  # executor decides
        if info["signal"] == "B":
            confirmed.append(info)
        elif info["signal"] == "C":
            candidates.append(info)

    candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    top_candidates = candidates[:MAX_CANDIDATES]

    signal_type = "HOLD"
    orders: List[dict] = []
    if confirmed:
        signal_type = "B"
        for o in confirmed:
            orders.append({
                "ticker": o["ticker"],
                "symbol": o["symbol"],
                "entry": round(o["entry"], 8),
                "stop": round(o["stop"], 8),
                "t1": round(o["t1"], 8),
                "t2": round(o["t2"], 8),
                "atr": round(o["atr"], 8),
                "tf": "1h",
                "notes": o.get("reasons", []),
                "rotation_exempt": False,
            })
    elif top_candidates:
        signal_type = "C"

    cand_payload = [{
        "ticker": c["ticker"],
        "symbol": c["symbol"],
        "last": round(c["last"], 8),
        "atr": round(c["atr"], 8),
        "entry": round(c["entry"], 8),
        "stop": round(c["stop"], 8),
        "t1": round(c["t1"], 8),
        "t2": round(c["t2"], 8),
        "score": round(c["score"], 4),
        "vol_z": round(c["vol_z"], 2),
        "prox_atr": round(c["prox_atr"], 3),
        "rs10": round(c["rs10"], 4),
        "tf": "1h",
        "notes": c.get("reasons", []),
        "rotation_exempt": False,
    } for c in top_candidates]

    payload: Dict[str, Any] = {
        "generated_at": human_time(started),
        "timezone": TZ_NAME,
        "regime": {"ok": regime_ok, "reason": regime_reason},
        "signals": {"type": signal_type},
        "orders": orders,
        "candidates": cand_payload,
        "universe": {
            "scanned": scanned,
            "eligible": len(mapping),
            "skipped": {
                "no_data": sorted(no_data),
                "missing_binance": [],  # always empty by design
            },
        },
        "meta": {
            "params": {
                "ATR_LEN": ATR_LEN, "EMA_FAST": EMA_FAST, "EMA_SLOW": EMA_SLOW,
                "SWING_LOOKBACK": SWING_LOOKBACK,
                "PROX_ATR_MIN": PROX_ATR_MIN, "PROX_ATR_MAX": PROX_ATR_MAX,
                "VOL_Z_MIN_PRE": VOL_Z_MIN_PRE, "VOL_Z_MIN_BREAK": VOL_Z_MIN_BREAK,
                "BREAK_BUFFER_ATR": BREAK_BUFFER_ATR,
                "MAX_CANDIDATES": MAX_CANDIDATES
            },
            "lower_tf_confirm": True,
            "rs_reference": "BTCUSDC",
            "binance_endpoint": BASE_URL,
            "mode": mode,
        }
    }

    return payload

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyses pipeline")
    parser.add_argument("--mode", default="deep", choices=["deep", "light-fast", "light-hourly"])
    args = parser.parse_args()

    try:
        payload = run_pipeline(args.mode)
    except Exception:
        log("UNCAUGHT ERROR:\n" + traceback.format_exc())
        payload = {
            "generated_at": human_time(),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "uncaught error"},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": 0, "eligible": 0, "skipped": {"no_data": [], "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": args.mode},
        }

    latest_path = Path("public_runs/latest/summary.json")
    write_summary(payload, latest_path)

    stamp = datetime.now(timezone.utc).astimezone(TZ).strftime("%Y%m%d_%H%M%S")
    snapshot_dir = Path("public_runs") / stamp
    ensure_dirs(snapshot_dir / "summary.json")
    with (snapshot_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

    log(f"Summary written to {latest_path} (signal={payload.get('signals',{}).get('type')}, regime_ok={payload.get('regime',{}).get('ok')})")

if __name__ == "__main__":
    main()
