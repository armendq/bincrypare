#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses pipeline - USDC spot scanner for pre-breakouts and breakouts.

Key points:
- Universe: all Binance Spot pairs with quoteAsset == USDC (excludes stables, leveraged suffixes).
- Timeframes: 4h/1h for trend; 15m/5m for momentum confirm.
- Signals:
  - "B" confirmed breakout above HH20 with volume thrust.
  - "C" pre-breakout near HH20 with momentum and volume.
  - "HOLD" otherwise.
- Regime: BTCUSDC uptrend filter, overridable with --ignore_regime or RELAX flags.
- Output: public_runs/latest/summary.json and timestamped copy under public_runs/YYYYmmdd_HHMMSS/summary.json
- Missing list is always [] by design (we build the universe from Binance exchangeInfo).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# ----------------------------- CONFIG ---------------------------------

TZ_NAME = "Europe/Prague"
TZ = ZoneInfo(TZ_NAME) if ZoneInfo else timezone.utc

# Endpoints
BASE_URL = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")
EXINFO_URL = f"{BASE_URL}/api/v3/exchangeInfo"
KLINES_URL = f"{BASE_URL}/api/v3/klines"

# Cache
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp")
EXINFO_CACHE_PATH = Path(CACHE_DIR) / "binance_exinfo_usdc.json"
EXINFO_TTL_SEC = int(os.getenv("EXINFO_TTL_SEC", "1800"))  # 30 min

# Intervals & fetch limits
INTERVALS = {
    "4h": ("4h", 500),
    "1h": ("1h", 600),
    "15m": ("15m", 400),
    "5m": ("5m", 400),
}

# Indicators
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "200"))
SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "20"))  # HH20/LL20

# Heuristics (env-overridable)
PROX_ATR_MIN = float(os.getenv("PROX_ATR_MIN", "0.05"))
PROX_ATR_MAX = float(os.getenv("PROX_ATR_MAX", "0.35"))
VOL_Z_MIN_PRE = float(os.getenv("VOL_Z_MIN_PRE", "1.2"))
VOL_Z_MIN_BREAK = float(os.getenv("VOL_Z_MIN_BREAK", "1.5"))
BREAK_BUFFER_ATR = float(os.getenv("BREAK_BUFFER_ATR", "0.06"))
RELAX_B_HIGH = os.getenv("RELAX_B_HIGH", "0") == "1"  # confirm with 1h HIGH instead of close

# Liquidity floor (approx USDC turnover on 1h)
MIN_AVG_VOL = float(os.getenv("MIN_AVG_VOL", "4000"))

# Position sizing advisory (not traded here; exec layer sizes for real)
CAPITAL = float(os.getenv("CAPITAL", "10000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MIN_RISK_FLOOR_PCT = float(os.getenv("MIN_RISK_FLOOR_PCT", "0.01"))
MAX_RISK_CAP_PCT = float(os.getenv("MAX_RISK_CAP_PCT", "0.04"))
STOP_ATR_BUFFER = float(os.getenv("STOP_ATR_BUFFER", "0.4"))

# Relax switch to increase signal frequency without editing code
RELAXED_SIGNALS = os.getenv("RELAXED_SIGNALS", "0") == "1"

# Ranking
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "10"))

# Universe filters
STABLES = {"USDT", "USDC", "DAI", "TUSD", "USDP", "BUSD", "FDUSD", "PYUSD"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S")

# Parallelism and pacing
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "12"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="[%(asctime)s] %(message)s")

# -------------------------- HELPERS -----------------------------------

def human_time(ts: Optional[datetime] = None) -> str:
    dt = ts or datetime.now(timezone.utc).astimezone(TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def safe_get(url: str, params: Optional[dict] = None, retries: int = 3, timeout: int = REQUEST_TIMEOUT, sleep_s: float = 0.6) -> Optional[requests.Response]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (418, 429, 451, 403, 500, 504):
                time.sleep(sleep_s * attempt)
            else:
                time.sleep(sleep_s * 0.5)
        except requests.RequestException:
            time.sleep(sleep_s * attempt)
    return None

def ema(series: np.ndarray, period: int) -> np.ndarray:
    if len(series) == 0 or period <= 1:
        return series
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(series, dtype=np.float64)
    out[0] = float(series[0])
    for i in range(1, len(series)):
        out[i] = alpha * float(series[i]) + (1 - alpha) * out[i - 1]
    return out

def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = ATR_LEN) -> np.ndarray:
    if len(highs) < 2:
        return np.zeros_like(highs, dtype=np.float64)
    prev_close = closes[:-1]
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - prev_close)
    tr3 = np.abs(lows[1:] - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, 0.0)  # align lengths
    return ema(tr, period)

def swing_high(series: np.ndarray, lookback: int) -> np.ndarray:
    return np.array([np.max(series[max(0, i - lookback + 1): i + 1]) for i in range(len(series))], dtype=np.float64)

def swing_low(series: np.ndarray, lookback: int) -> np.ndarray:
    return np.array([np.min(series[max(0, i - lookback + 1): i + 1]) for i in range(len(series))], dtype=np.float64)

def pct_change(series: np.ndarray, n: int) -> float:
    if len(series) < n + 1:
        return 0.0
    a, b = float(series[-n - 1]), float(series[-1])
    if a == 0:
        return 0.0
    return (b - a) / a

def vol_zscore(vols: np.ndarray, window: int = 50) -> float:
    if len(vols) < window + 1:
        return 0.0
    sample = vols[-(window + 1):-1]
    mu = float(np.mean(sample))
    sd = float(np.std(sample, ddof=0))
    if sd == 0:
        return 0.0
    return (float(vols[-1]) - mu) / sd

# -------------------------- DATA SOURCING ------------------------------

def load_exinfo_cache() -> Optional[dict]:
    try:
        if EXINFO_CACHE_PATH.exists():
            mtime = EXINFO_CACHE_PATH.stat().st_mtime
            if (time.time() - mtime) <= EXINFO_TTL_SEC:
                return json.loads(EXINFO_CACHE_PATH.read_text("utf-8"))
    except Exception:
        pass
    return None

def save_exinfo_cache(data: dict) -> None:
    try:
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        EXINFO_CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        logging.warning("Cache write failed; proceeding without cache.")

def fetch_exchange_usdc_symbols() -> Dict[str, dict]:
    data = load_exinfo_cache()
    if data is None:
        r = safe_get(EXINFO_URL, retries=3, timeout=25)
        if not r:
            return {}
        data = r.json()
        save_exinfo_cache(data)
    out: Dict[str, dict] = {}
    for s in data.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != "USDC":
            continue
        if not s.get("isSpotTradingAllowed", True):
            continue
        base = (s.get("baseAsset") or "").upper()
        if not base:
            continue
        if base in STABLES:
            continue
        if any(base.endswith(sfx) for sfx in LEVERAGED_SUFFIXES):
            continue
        out[s["symbol"]] = s
    return out

def fetch_klines(symbol: str, interval: str, limit: int, end_time: Optional[int] = None) -> Optional[List[List[Any]]]:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
        params["endTime"] = end_time
    r = safe_get(KLINES_URL, params=params, retries=3, timeout=REQUEST_TIMEOUT)
    if not r:
        return None
    try:
        data = r.json()
        if isinstance(data, list):
            return data
    except Exception:
        return None
    return None

def fetch_all_klines(symbols: List[str], interval: str, limit: int, end_time: Optional[int] = None) -> Dict[str, Optional[List[List[Any]]]]:
    results: Dict[str, Optional[List[List[Any]]]] = {}
    # shuffle to spread symbols to different minutes over time
    idxs = list(range(len(symbols)))
    random.shuffle(idxs)
    shuffled = [symbols[i] for i in idxs]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut = {ex.submit(fetch_klines, sym, interval, limit, end_time): sym for sym in shuffled}
        for f in as_completed(fut):
            sym = fut[f]
            try:
                results[sym] = f.result()
            except Exception as e:
                logging.warning(f"kline fetch failed {sym}: {e}")
                results[sym] = None
    return results

def parse_klines(raw: List[List[Any]]) -> Dict[str, np.ndarray]:
    return {
        "open_time": np.array([int(r[0]) for r in raw], dtype=np.int64),
        "open": np.array([float(r[1]) for r in raw], dtype=np.float64),
        "high": np.array([float(r[2]) for r in raw], dtype=np.float64),
        "low": np.array([float(r[3]) for r in raw], dtype=np.float64),
        "close": np.array([float(r[4]) for r in raw], dtype=np.float64),
        "volume": np.array([float(r[5]) for r in raw], dtype=np.float64),
    }

# -------------------------- CORE LOGIC --------------------------------

def analyze_symbol(symbol: str, req: Dict[str, Dict[str, np.ndarray]], btc_1h_close: np.ndarray) -> Optional[dict]:
    try:
        one = req["1h"]; four = req["4h"]; fifteen = req["15m"]; five = req["5m"]
        close_1h = one["close"]; high_1h = one["high"]; low_1h = one["low"]; vol_1h = one["volume"]

        need = max(EMA_SLOW + 5, SWING_LOOKBACK + 5)
        if len(close_1h) < need or len(four["close"]) < need:
            return None

        # Trend filters
        ema20_1h = ema(close_1h, EMA_FAST)
        ema200_1h = ema(close_1h, EMA_SLOW)
        ema20_4h = ema(four["close"], EMA_FAST)
        ema200_4h = ema(four["close"], EMA_SLOW)

        ema20_slope_1h = ema20_1h[-1] - ema20_1h[-4]
        ema20_slope_4h = ema20_4h[-1] - ema20_4h[-4]
        trend_ok = (ema20_slope_1h > 0 and ema20_slope_4h > 0 and close_1h[-1] > ema200_1h[-1] and four["close"][-1] > ema200_4h[-1])

        # ATR & swings on 1h
        atr_1h = atr(high_1h, low_1h, close_1h, ATR_LEN)
        if atr_1h[-1] <= 0:
            return None
        hh20 = swing_high(high_1h, SWING_LOOKBACK)
        ll20 = swing_low(low_1h, SWING_LOOKBACK)

        # Volume context
        vz = vol_zscore(vol_1h, window=50)

        # Liquidity filter (approx USDC turnover)
        avg_vol_usdc = float(np.mean(vol_1h[-10:]) * close_1h[-1])
        if avg_vol_usdc < MIN_AVG_VOL:
            return None

        # Lower TF momentum
        def mom_ok(tf: Dict[str, np.ndarray]) -> bool:
            c = tf["close"]; e20 = ema(c, EMA_FAST)
            if len(e20) < 5:
                return False
            return (c[-1] > e20[-1]) and (e20[-1] > e20[-3])

        mom15 = mom_ok(fifteen); mom5 = mom_ok(five)
        lower_tf_ok = mom15 and mom5

        # RS vs BTC (1h)
        rs_strength = 0.0
        if len(btc_1h_close) == len(close_1h):
            rs_series = close_1h / np.where(btc_1h_close != 0, btc_1h_close, 1e-9)
            rs_strength = pct_change(rs_series, n=10)

        last_close = close_1h[-1]
        last_high = high_1h[-1]
        last_low = low_1h[-1]
        last_atr = atr_1h[-1]
        breakout_level = hh20[-1]

        # Reasons & score
        reasons: List[str] = []
        if trend_ok: reasons.append("TrendOK(1h&4h)")
        if vz > 0: reasons.append(f"VolZ={vz:.2f}")
        if mom15: reasons.append("15m OK")
        if mom5: reasons.append("5m OK")

        score = 0.0
        if vz > 0: score += vz
        prox = (breakout_level - last_close) / max(1e-9, last_atr)
        if prox > 0: score += 1.0 / (min(max(prox, 0.01), 2.0))
        score += max(0.0, 5.0 * rs_strength)
        if lower_tf_ok: score += 0.5
        score += np.log1p(avg_vol_usdc / 1e5)

        # Relax switches
        confirm_price = last_high if RELAX_B_HIGH else last_close
        vz_break = VOL_Z_MIN_BREAK if not RELAXED_SIGNALS else max(0.8, VOL_Z_MIN_BREAK - 0.2)

        confirmed_breakout = (
            (trend_ok or (RELAXED_SIGNALS and lower_tf_ok)) and
            confirm_price > (breakout_level + BREAK_BUFFER_ATR * last_atr) and
            vz >= vz_break and
            (lower_tf_ok or RELAXED_SIGNALS)
        )

        prox_min = PROX_ATR_MIN
        prox_max = PROX_ATR_MAX
        vz_pre = VOL_Z_MIN_PRE
        if RELAXED_SIGNALS:
            prox_min = min(-0.15, PROX_ATR_MIN)
            prox_max = max(0.60, PROX_ATR_MAX)
            vz_pre = min(1.0, VOL_Z_MIN_PRE)

        pre_breakout = (
            (trend_ok or (RELAXED_SIGNALS and (lower_tf_ok or vz >= 0.8))) and
            (prox_min <= prox <= prox_max) and
            vz >= vz_pre
        )

        # Levels
        entry = breakout_level + BREAK_BUFFER_ATR * last_atr
        # widen stop below LL20 with buffer, and cap/floor risk in percent terms
        base_stop = float(ll20[-1]) - STOP_ATR_BUFFER * last_atr
        natural_risk = max(entry - base_stop, 0.0)
        min_risk_width = max(MIN_RISK_FLOOR_PCT * entry, entry * 0.002)
        max_risk_width = MAX_RISK_CAP_PCT * entry
        entry_to_stop = min(max(natural_risk, min_risk_width), max_risk_width)
        stop = entry - entry_to_stop

        t1 = entry + 0.8 * last_atr
        t2 = entry + 1.5 * last_atr

        # Advisory position sizing (for monitoring only)
        position_size = 0.0
        position_size_usdc = 0.0
        if entry > 0 and entry_to_stop > 0:
            risk_dollars = CAPITAL * RISK_PER_TRADE
            units = max(risk_dollars / (entry_to_stop), 0.0)
            position_size = units
            position_size_usdc = units * entry

        out: Dict[str, Any] = {
            "symbol": symbol,
            "last": float(last_close),
            "atr": float(last_atr),
            "hh20": float(breakout_level),
            "ll20": float(ll20[-1]),
            "entry": float(entry),
            "stop": float(stop),
            "t1": float(t1),
            "t2": float(t2),
            "vol_z": float(vz),
            "prox_atr": float(prox),
            "trend_ok": bool(trend_ok),
            "lower_tf_ok": bool(lower_tf_ok),
            "rs10": float(rs_strength),
            "score": float(score),
            "reasons": reasons,
            "avg_vol": float(avg_vol_usdc),
            "position_size": float(position_size),
            "position_size_usdc": float(position_size_usdc),
        }

        if confirmed_breakout:
            out["signal"] = "B"
        elif pre_breakout:
            out["signal"] = "C"
        else:
            out["signal"] = "N"

        return out
    except Exception as e:
        logging.warning(f"analysis failed {symbol}: {e}")
        return None

def write_summary(payload: dict, dest_latest: Path) -> None:
    ensure_dirs(dest_latest)
    with dest_latest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

# -------------------------- PIPELINE ----------------------------------

def build_universe() -> Tuple[Dict[str, str], List[str]]:
    exch = fetch_exchange_usdc_symbols()
    if not exch:
        return {}, []
    mapping: Dict[str, str] = {}
    for sym, meta in exch.items():
        base = (meta.get("baseAsset") or "").upper()
        if not base:
            continue
        mapping[base] = sym
    return mapping, []  # missing list intentionally empty

def compute_regime(btc_1h: Dict[str, np.ndarray], btc_4h: Dict[str, np.ndarray]) -> Tuple[bool, str]:
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

def process(symbols: List[str], mode: str, ignore_regime: bool, end_time: Optional[int] = None) -> dict:
    started = datetime.now(timezone.utc).astimezone(TZ) if not end_time else datetime.fromtimestamp(end_time / 1000, timezone.utc).astimezone(TZ)

    # Baseline: BTCUSDC
    btc_symbol = "BTCUSDC"
    btc_raw_1h = fetch_klines(btc_symbol, INTERVALS["1h"][0], INTERVALS["1h"][1], end_time)
    btc_raw_4h = fetch_klines(btc_symbol, INTERVALS["4h"][0], INTERVALS["4h"][1], end_time)

    if not btc_raw_1h or not btc_raw_4h:
        return {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "BTC data unavailable"},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": 0, "eligible": len(symbols), "skipped": {"no_data": [], "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode},
        }

    btc_1h = parse_klines(btc_raw_1h)
    btc_4h = parse_klines(btc_raw_4h)
    regime_ok, regime_reason = compute_regime(btc_1h, btc_4h)
    if ignore_regime:
        regime_ok = True
        regime_reason = "Regime ignored by flag"
    btc_close_1h = btc_1h["close"]

    # Mode thinning
    pool = symbols[:]
    if mode == "light-fast":
        random.shuffle(pool)
        pool = pool[: max(1, len(pool) // 2)]
    elif mode == "light-hourly":
        random.shuffle(pool)
        pool = pool[: max(1, len(pool) // 3)]

    # Parallel fetch TFs
    reqs: Dict[str, Dict[str, Optional[Dict[str, np.ndarray]]]] = {}
    for tf, (interval, limit) in INTERVALS.items():
        raw_all = fetch_all_klines(pool, interval, limit, end_time)
        reqs[tf] = {sym: (parse_klines(raw_all[sym]) if raw_all.get(sym) else None) for sym in pool}

    candidates: List[dict] = []
    confirmed: List[dict] = []
    no_data: List[str] = []
    scanned = 0

    for sym in pool:
        scanned += 1
        tf_data = {tf: reqs[tf][sym] for tf in INTERVALS}
        if any(d is None for d in tf_data.values()):
            no_data.append(sym.split("USDC")[0])
            continue
        info = analyze_symbol(sym, tf_data, btc_close_1h)
        if not info:
            no_data.append(sym.split("USDC")[0])
            continue
        info["ticker"] = sym.split("USDC")[0]
        info["rotation_exempt"] = False
        if info["signal"] == "B":
            confirmed.append(info)
        elif info["signal"] == "C":
            candidates.append(info)

    if not regime_ok:
        return {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": regime_ok, "reason": regime_reason},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": scanned, "eligible": len(symbols), "skipped": {"no_data": sorted(no_data), "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode},
        }

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
                "entry": round(float(o["entry"]), 8),
                "stop": round(float(o["stop"]), 8),
                "t1": round(float(o["t1"]), 8),
                "t2": round(float(o["t2"]), 8),
                "atr": round(float(o["atr"]), 8),
                "tf": "1h",
                "notes": o.get("reasons", []),
                "rotation_exempt": False,
                "position_size": round(float(o["position_size"]), 8),
                "position_size_usdc": round(float(o["position_size_usdc"]), 2),
            })
    elif top_candidates:
        signal_type = "C"

    cand_payload = [{
        "ticker": c["ticker"],
        "symbol": c["symbol"],
        "last": round(float(c["last"]), 8),
        "atr": round(float(c["atr"]), 8),
        "entry": round(float(c["entry"]), 8),
        "stop": round(float(c["stop"]), 8),
        "t1": round(float(c["t1"]), 8),
        "t2": round(float(c["t2"]), 8),
        "score": round(float(c["score"]), 4),
        "vol_z": round(float(c["vol_z"]), 2),
        "prox_atr": round(float(c["prox_atr"]), 3),
        "rs10": round(float(c["rs10"]), 4),
        "tf": "1h",
        "notes": c.get("reasons", []),
        "rotation_exempt": False,
        "position_size": round(float(c["position_size"]), 8),
        "position_size_usdc": round(float(c["position_size_usdc"]), 2),
    } for c in top_candidates]

    payload: Dict[str, Any] = {
        "generated_at": human_time(started),
        "timezone": TZ_NAME,
        "regime": {"ok": bool(regime_ok), "reason": regime_reason},
        "signals": {"type": signal_type},
        "orders": orders,
        "candidates": cand_payload,
        "universe": {
            "scanned": scanned,
            "eligible": len(symbols),
            "skipped": {"no_data": sorted(no_data), "missing_binance": []},
        },
        "meta": {
            "params": {
                "ATR_LEN": ATR_LEN, "EMA_FAST": EMA_FAST, "EMA_SLOW": EMA_SLOW,
                "SWING_LOOKBACK": SWING_LOOKBACK,
                "PROX_ATR_MIN": PROX_ATR_MIN, "PROX_ATR_MAX": PROX_ATR_MAX,
                "VOL_Z_MIN_PRE": VOL_Z_MIN_PRE, "VOL_Z_MIN_BREAK": VOL_Z_MIN_BREAK,
                "BREAK_BUFFER_ATR": BREAK_BUFFER_ATR,
                "MAX_CANDIDATES": MAX_CANDIDATES,
                "MIN_AVG_VOL": MIN_AVG_VOL,
                "CAPITAL": CAPITAL,
                "RISK_PER_TRADE": RISK_PER_TRADE,
                "RELAXED_SIGNALS": RELAXED_SIGNALS,
                "STOP_ATR_BUFFER": STOP_ATR_BUFFER,
                "MIN_RISK_FLOOR_PCT": MIN_RISK_FLOOR_PCT,
                "MAX_RISK_CAP_PCT": MAX_RISK_CAP_PCT,
            },
            "lower_tf_confirm": True,
            "rs_reference": "BTCUSDC",
            "binance_endpoint": BASE_URL,
            "mode": mode,
        }
    }
    return payload

def run_pipeline(mode: str, ignore_regime: bool = False, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
    mapping, _ = build_universe()
    if not mapping:
        return {
            "generated_at": human_time(),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "exchangeInfo unavailable or no eligible symbols"},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": 0, "eligible": 0, "skipped": {"no_data": [], "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode},
        }

    symbols = list(mapping.values())

    # Backtest mode (kept for completeness; not used in live workflow)
    if start_date and end_date:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        current_dt = start_dt
        all_payloads: List[dict] = []
        while current_dt <= end_dt:
            end_time_ms = int(current_dt.timestamp() * 1000)
            payload = process(symbols, mode, ignore_regime, end_time=end_time_ms)
            all_payloads.append(payload)
            current_dt += timedelta(days=1)
        stats = {
            "total_days": len(all_payloads),
            "total_breakouts": sum(len(p.get("orders", [])) for p in all_payloads),
            "avg_candidates": float(statistics.fmean(len(p.get("candidates", [])) for p in all_payloads)) if all_payloads else 0.0,
        }
        return {"backtest_results": all_payloads, "stats": stats}

    return process(symbols, mode, ignore_regime, end_time=None)

# -------------------------- MAIN --------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyses pipeline (USDC)")
    parser.add_argument("--mode", default="deep", choices=["deep", "light-fast", "light-hourly"])
    parser.add_argument("--ignore_regime", action="store_true", help="Ignore BTC regime check")
    parser.add_argument("--backtest", action="store_true", help="Backtest mode across days")
    parser.add_argument("--start_date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, help="YYYY-MM-DD")
    args = parser.parse_args()

    if args.backtest and (not args.start_date or not args.end_date):
        logging.error("Backtest requires --start_date and --end_date")
        return

    try:
        if args.backtest:
            payload = run_pipeline(args.mode, args.ignore_regime, args.start_date, args.end_date)
        else:
            payload = run_pipeline(args.mode, args.ignore_regime)
    except Exception:
        logging.error("UNCAUGHT ERROR:\n" + traceback.format_exc())
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

    logging.info(f"Summary written to {latest_path} (signal={payload.get('signals',{}).get('type')}, regime_ok={payload.get('regime',{}).get('ok')})")

if __name__ == "__main__":
    main()