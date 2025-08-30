#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses pipeline — scanner for pre-breakouts and breakouts (USDC universe)

- Universe: all Binance Spot pairs with quoteAsset == USDC
- Excludes bases that are stablecoins and leveraged/multiplier tokens
- BTC reference: BTCUSDC
- Emits public_runs/latest/summary.json and a timestamped copy
- missing_binance is always [] (we only use Binance symbols)
- Adds RELAXED_SIGNALS path for quiet markets (env toggle)
- Improved stop sizing (buffer under LL20, min/max risk rails)
- Near-miss logging + payload meta for diagnostics
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
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

# ----------------------------- CONFIG ---------------------------------

TZ_NAME = os.getenv("TZ_NAME", "Europe/Prague")
TZ = ZoneInfo(TZ_NAME) if ZoneInfo else timezone.utc

BASE_URL = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")

# Intervals & limits
INTERVALS = {
    "4h": ("4h", 500),
    "1h": ("1h", 600),
    "15m": ("15m", 400),
    "5m": ("5m", 400),
}

# ATR / EMA parameters
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "200"))
SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "20"))

# Heuristics (env-overridable)
PROX_ATR_MIN = float(os.getenv("PROX_ATR_MIN", "0.05"))
PROX_ATR_MAX = float(os.getenv("PROX_ATR_MAX", "0.35"))
VOL_Z_MIN_PRE = float(os.getenv("VOL_Z_MIN_PRE", "1.2"))
VOL_Z_MIN_BREAK = float(os.getenv("VOL_Z_MIN_BREAK", "1.5"))
BREAK_BUFFER_ATR = float(os.getenv("BREAK_BUFFER_ATR", "0.06"))
RELAX_B_HIGH = os.getenv("RELAX_B_HIGH", "1") == "1"   # use 1h HIGH instead of close to confirm

# Liquidity / risk rails
MIN_AVG_VOL = float(os.getenv("MIN_AVG_VOL", "1000"))  # approx USDC over last 10x 1h bars
STOP_ATR_BUFFER = float(os.getenv("STOP_ATR_BUFFER", "0.4"))  # how far below LL20 we park stop (in ATRs)
MIN_RISK_FLOOR_PCT = float(os.getenv("MIN_RISK_FLOOR_PCT", "0.008"))  # >=0.8% width
MAX_RISK_CAP_PCT = float(os.getenv("MAX_RISK_CAP_PCT", "0.05"))       # <=5% width
CAPITAL = float(os.getenv("CAPITAL", "10000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))

# Relaxed mode (for quiet markets)
RELAXED_SIGNALS = os.getenv("RELAXED_SIGNALS", "0") == "1"
RELAXED_VZ_PRE_FLOOR = float(os.getenv("RELAXED_VZ_PRE_FLOOR", "0.5"))
RELAXED_VZ_BREAK_FLOOR = float(os.getenv("RELAXED_VZ_BREAK_FLOOR", "0.7"))
RELAXED_PROX_MIN_FLOOR = float(os.getenv("RELAXED_PROX_MIN_FLOOR", "-0.10"))
RELAXED_PROX_MAX_CEIL = float(os.getenv("RELAXED_PROX_MAX_CEIL", "1.0"))

# Ranking & threads
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "10"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "12"))

# Stable/leveraged filters
STABLES = {"USDT", "USDC", "DAI", "TUSD", "USDP", "BUSD", "FDUSD", "PYUSD"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S")

# Diagnostics
NEAR_MISS_LOG_COUNT = int(os.getenv("NEAR_MISS_LOG_COUNT", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="[%(asctime)s] %(message)s")

# -------------------------- HELPERS -----------------------------------

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

def ema(series: np.ndarray, period: int) -> np.ndarray:
    if len(series) == 0 or period <= 1:
        return series
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out

def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = ATR_LEN) -> np.ndarray:
    if len(highs) < 2:
        return np.zeros_like(highs)
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate(([0.0], tr))
    return ema(tr, period)

def swing_high(series: np.ndarray, lookback: int) -> np.ndarray:
    return np.array([np.max(series[max(0, i - lookback + 1): i + 1]) for i in range(len(series))])

def swing_low(series: np.ndarray, lookback: int) -> np.ndarray:
    return np.array([np.min(series[max(0, i - lookback + 1): i + 1]) for i in range(len(series))])

def pct_change(series: np.ndarray, n: int) -> float:
    if len(series) < n + 1:
        return 0.0
    a, b = series[-n - 1], series[-1]
    if a == 0:
        return 0.0
    return (b - a) / a

def vol_zscore(vols: np.ndarray, window: int = 50) -> float:
    if len(vols) < window + 1:
        return 0.0
    sample = vols[-(window + 1):-1]
    mu = float(np.mean(sample))
    sd = float(np.std(sample))
    if sd == 0.0:
        return 0.0
    return float((vols[-1] - mu) / sd)

def fetch_exchange_usdc_symbols() -> Dict[str, dict]:
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

def fetch_klines(symbol: str, interval: str, limit: int, end_time: Optional[int] = None) -> Optional[List[List[Any]]]:
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time is not None:
        params["endTime"] = end_time
    r = safe_get(url, params=params, retries=3, timeout=20)
    if not r:
        return None
    try:
        data = r.json()
        return data if isinstance(data, list) else None
    except Exception:
        return None

def fetch_all_klines(symbols: List[str], interval: str, limit: int, end_time: Optional[int] = None) -> Dict[str, Optional[List[List[Any]]]]:
    results: Dict[str, Optional[List[List[Any]]]] = {}
    if not symbols:
        return results
    with ThreadPoolExecutor(max_workers=max(1, min(MAX_WORKERS, len(symbols)))) as pool:
        futs = {pool.submit(fetch_klines, sym, interval, limit, end_time): sym for sym in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                results[sym] = fut.result()
            except Exception as e:
                logging.warning(f"Failed to fetch {interval} for {sym}: {e}")
                results[sym] = None
    return results

def parse_klines(raw: List[List[Any]]) -> Dict[str, np.ndarray]:
    return {
        "open_time": np.array([int(r[0]) for r in raw]),
        "open": np.array([float(r[1]) for r in raw]),
        "high": np.array([float(r[2]) for r in raw]),
        "low": np.array([float(r[3]) for r in raw]),
        "close": np.array([float(r[4]) for r in raw]),
        "volume": np.array([float(r[5]) for r in raw]),
    }

# -------------------------- CORE LOGIC --------------------------------

def analyze_symbol(symbol: str, btc_1h_close: np.ndarray, req: Dict[str, Dict[str, np.ndarray]]) -> Optional[dict]:
    try:
        one = req["1h"]; four = req["4h"]; fifteen = req["15m"]; five = req["5m"]

        close_1h = one["close"]; high_1h = one["high"]; low_1h = one["low"]; vol_1h = one["volume"]
        if len(close_1h) < max(EMA_SLOW + 5, SWING_LOOKBACK + 5):
            return None

        # Trends
        ema20_1h = ema(close_1h, EMA_FAST)
        ema200_1h = ema(close_1h, EMA_SLOW)
        ema20_4h = ema(four["close"], EMA_FAST)
        ema200_4h = ema(four["close"], EMA_SLOW)

        ema20_slope_1h = float(ema20_1h[-1] - ema20_1h[-4])
        ema20_slope_4h = float(ema20_4h[-1] - ema20_4h[-4])

        trend_ok_strict = (
            ema20_slope_1h > 0 and ema20_slope_4h > 0 and
            close_1h[-1] > ema200_1h[-1] and four["close"][-1] > ema200_4h[-1]
        )
        # relaxed trend: price above its EMA20s and EMA20 slopes up (no 200 filter)
        trend_ok_relaxed = (
            ema20_slope_1h > 0 and ema20_slope_4h > 0 and
            close_1h[-1] > ema20_1h[-1] and four["close"][-1] > ema20_4h[-1]
        )
        trend_ok = trend_ok_strict or (RELAXED_SIGNALS and trend_ok_relaxed)

        # ATR, swings
        atr_1h = atr(high_1h, low_1h, close_1h, ATR_LEN)
        if float(atr_1h[-1]) <= 0:
            return None
        hh20 = swing_high(high_1h, SWING_LOOKBACK)
        ll20 = swing_low(low_1h, SWING_LOOKBACK)

        # Volume context & liquidity
        vz = vol_zscore(vol_1h, window=50)
        avg_vol = float(np.mean(vol_1h[-10:]) * close_1h[-1])  # approx USDC
        if avg_vol < MIN_AVG_VOL:
            return None

        # Momentum confirms
        def mom_ok(tf_data: Dict[str, np.ndarray]) -> bool:
            c = tf_data["close"]; e20 = ema(c, EMA_FAST)
            if len(e20) < 5: return False
            return bool((c[-1] > e20[-1]) and (e20[-1] > e20[-3]))

        mom15 = mom_ok(fifteen); mom5 = mom_ok(five)
        lower_tf_ok = mom15 and mom5

        # RS vs BTC
        rs_strength = 0.0
        if len(btc_1h_close) == len(close_1h):
            base = np.where(btc_1h_close != 0, btc_1h_close, 1e-9)
            rs_series = close_1h / base
            rs_strength = pct_change(rs_series, n=10)

        last_close = float(close_1h[-1])
        last_high = float(high_1h[-1])
        last_low = float(low_1h[-1])
        last_atr = float(atr_1h[-1])

        # Proximity (ATRs below HH20)
        prox = float((hh20[-1] - close_1h[-1]) / max(1e-9, last_atr))
        atr_rising = bool(atr_1h[-1] > atr_1h[-2] > atr_1h[-3])

        # Effective thresholds (can relax)
        prox_min = PROX_ATR_MIN
        prox_max = PROX_ATR_MAX
        vz_pre = VOL_Z_MIN_PRE
        vz_break = VOL_Z_MIN_BREAK
        if RELAXED_SIGNALS:
            prox_min = min(prox_min, RELAXED_PROX_MIN_FLOOR)
            prox_max = max(prox_max, RELAXED_PROX_MAX_CEIL)
            vz_pre = min(vz_pre, RELAXED_VZ_PRE_FLOOR)
            vz_break = min(vz_break, RELAXED_VZ_BREAK_FLOOR)

        # Score & reasons
        reasons: List[str] = []
        if trend_ok: reasons.append("TrendOK")
        if atr_rising: reasons.append("ATR up")
        if vz > 0: reasons.append(f"VolZ={vz:.2f}")
        if lower_tf_ok: reasons.append("LowerTF OK")
        if rs_strength > 0: reasons.append(f"RS={rs_strength:.2f}")

        score = 0.0
        if vz > 0: score += vz
        if prox > 0: score += 1.0 / (min(max(prox, 0.01), 2.0))
        score += max(0.0, 5.0 * rs_strength)
        if lower_tf_ok: score += 0.5
        score += float(np.log1p(max(0.0, avg_vol) / 1e5))  # mild liquidity boost

        # Breakout / pre-breakout
        breakout_level = float(hh20[-1])
        confirm_price = last_high if RELAX_B_HIGH else last_close

        confirmed_breakout = (
            trend_ok
            and confirm_price > (breakout_level + BREAK_BUFFER_ATR * last_atr)
            and vz >= vz_break
            and lower_tf_ok
        )
        pre_breakout = (
            trend_ok
            and atr_rising
            and (prox_min <= prox <= prox_max)
            and vz >= vz_pre
        )

        # Entry/targets
        entry = breakout_level + BREAK_BUFFER_ATR * last_atr

        # Stop sizing (buffer under LL20, risk floor/cap)
        base_stop = float(ll20[-1]) - STOP_ATR_BUFFER * last_atr
        natural_risk = max(1e-12, entry - base_stop)  # width
        min_risk_w = max(MIN_RISK_FLOOR_PCT * entry, 1e-12)
        max_risk_w = MAX_RISK_CAP_PCT * entry
        risk_w = min(max(natural_risk, min_risk_w), max_risk_w)
        stop = entry - risk_w

        t1 = entry + 0.8 * last_atr
        t2 = entry + 1.5 * last_atr

        # Position sizing (advisory)
        if risk_w > 0:
            position_size_usdc = (CAPITAL * RISK_PER_TRADE) / (risk_w / entry)
            position_size = position_size_usdc / entry
        else:
            position_size_usdc = 0.0
            position_size = 0.0

        out: Dict[str, Any] = {
            "symbol": symbol,
            "last": last_close,
            "atr": last_atr,
            "hh20": breakout_level,
            "ll20": float(ll20[-1]),
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
            "avg_vol": avg_vol,
            "position_size": position_size,
            "position_size_usdc": position_size_usdc,
        }

        if confirmed_breakout:
            out["signal"] = "B"
        elif pre_breakout:
            out["signal"] = "C"
        else:
            out["signal"] = "N"
            # diagnostics for near-miss
            miss = []
            if not trend_ok: miss.append("No TrendOK")
            if not lower_tf_ok: miss.append("No LowerTF")
            if vz < vz_pre: miss.append(f"Low vz={vz:.2f} (<{vz_pre})")
            if not (prox_min <= prox <= prox_max): miss.append(f"Prox out={prox:.2f}")
            out["miss_reasons"] = miss

        return out

    except Exception as e:
        logging.warning(f"Analysis failed for {symbol}: {e}")
        return None

def write_summary(payload: dict, dest_latest: Path) -> None:
    ensure_dirs(dest_latest)
    with dest_latest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

# -------------------------- MAIN PIPELINE -----------------------------

def build_universe() -> Tuple[Dict[str, str], List[str]]:
    exch = fetch_exchange_usdc_symbols()
    if not exch:
        return {}, []
    mapping: Dict[str, str] = {}
    for sym, meta in exch.items():
        base = meta.get("baseAsset", "").upper()
        if base:
            mapping[base] = sym
    return mapping, []

def compute_regime(btc_1h: Dict[str, np.ndarray], btc_4h: Dict[str, np.ndarray]) -> Tuple[bool, str]:
    try:
        ema200_4h = ema(btc_4h["close"], EMA_SLOW)
        ema20_4h = ema(btc_4h["close"], EMA_FAST)
        ema200_1h = ema(btc_1h["close"], EMA_SLOW)
        ema20_1h = ema(btc_1h["close"], EMA_FAST)
        if len(ema200_4h) > 4 and len(ema200_1h) > 4:
            slope4 = float(ema20_4h[-1] - ema20_4h[-4])
            slope1 = float(ema20_1h[-1] - ema20_1h[-4])
            if (btc_4h["close"][-1] > ema200_4h[-1]) and (btc_1h["close"][-1] > ema200_1h[-1]) and slope4 > 0 and slope1 > 0:
                return True, "BTC uptrend (4h & 1h)"
            return False, "BTC not in uptrend"
        return False, "insufficient data"
    except Exception:
        return False, "regime calc error"

def process(symbols: List[str], mode: str, ignore_regime: bool, end_time: Optional[int]) -> dict:
    started = datetime.now(timezone.utc).astimezone(TZ) if end_time is None else datetime.fromtimestamp(end_time / 1000, timezone.utc).astimezone(TZ)

    # RS baseline
    btc_raw_1h = fetch_klines("BTCUSDC", INTERVALS["1h"][0], INTERVALS["1h"][1], end_time)
    btc_raw_4h = fetch_klines("BTCUSDC", INTERVALS["4h"][0], INTERVALS["4h"][1], end_time)
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
    btc_1h = parse_klines(btc_raw_1h); btc_4h = parse_klines(btc_raw_4h)
    regime_ok, regime_reason = compute_regime(btc_1h, btc_4h)
    if ignore_regime:
        regime_ok = True
        regime_reason = "Regime ignored by flag"
    btc_close_1h = btc_1h["close"]

    # Thinning for light modes
    if mode == "light-fast":
        random.shuffle(symbols); symbols = symbols[: max(1, len(symbols)//2)]
    elif mode == "light-hourly":
        random.shuffle(symbols); symbols = symbols[: max(1, len(symbols)//3)]

    # Fetch all TFs in parallel
    reqs: Dict[str, Dict[str, Optional[Dict[str, np.ndarray]]]] = {}
    for tf, (interval, limit) in INTERVALS.items():
        raw_all = fetch_all_klines(symbols, interval, limit, end_time)
        reqs[tf] = {sym: (parse_klines(raw_all[sym]) if raw_all[sym] else None) for sym in symbols}

    candidates: List[dict] = []
    confirmed: List[dict] = []
    all_infos: List[dict] = []
    no_data: List[str] = []
    scanned = 0

    for sym in symbols:
        scanned += 1
        tf_data = {tf: reqs[tf].get(sym) for tf in INTERVALS}
        if any(d is None for d in tf_data.values()):
            no_data.append(sym.split("USDC")[0]); continue
        info = analyze_symbol(sym, btc_close_1h, tf_data)  # may be N/C/B
        if not info:
            no_data.append(sym.split("USDC")[0]); continue
        info["ticker"] = sym.split("USDC")[0]
        info["rotation_exempt"] = False
        all_infos.append(info)
        if info["signal"] == "B":
            confirmed.append(info)
        elif info["signal"] == "C":
            candidates.append(info)

    # If you want signals no matter what, keep regime filter only for B execution layer
    if not regime_ok:
        payload = {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": regime_ok, "reason": regime_reason},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": scanned, "eligible": len(symbols), "skipped": {"no_data": sorted(no_data), "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode},
        }
        return payload

    candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    top_candidates = candidates[:MAX_CANDIDATES]

    signal_type = "HOLD"; orders: List[dict] = []
    if confirmed:
        signal_type = "B"
        for o in confirmed:
            orders.append({
                "ticker": o["ticker"], "symbol": o["symbol"],
                "entry": round(o["entry"], 8), "stop": round(o["stop"], 8),
                "t1": round(o["t1"], 8), "t2": round(o["t2"], 8),
                "atr": round(o["atr"], 8), "tf": "1h",
                "notes": o.get("reasons", []), "rotation_exempt": False,
                "position_size": round(o["position_size"], 8),
                "position_size_usdc": round(o["position_size_usdc"], 2),
            })
    elif top_candidates:
        signal_type = "C"

    cand_payload = [{
        "ticker": c["ticker"], "symbol": c["symbol"],
        "last": round(c["last"], 8), "atr": round(c["atr"], 8),
        "entry": round(c["entry"], 8), "stop": round(c["stop"], 8),
        "t1": round(c["t1"], 8), "t2": round(c["t2"], 8),
        "score": round(c["score"], 4), "vol_z": round(c["vol_z"], 2),
        "prox_atr": round(c["prox_atr"], 3), "rs10": round(c["rs10"], 4),
        "tf": "1h", "notes": c.get("reasons", []), "rotation_exempt": False,
        "position_size": round(c["position_size"], 8),
        "position_size_usdc": round(c["position_size_usdc"], 2),
    } for c in top_candidates]

    # Near-miss diagnostics
    non_signals = [i for i in all_infos if i.get("signal") == "N"]
    non_signals.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    near_misses = [{
        "symbol": ns["symbol"],
        "score": round(ns.get("score", 0.0), 3),
        "vz": round(ns.get("vol_z", 0.0), 2),
        "prox": round(ns.get("prox_atr", 0.0), 3),
        "miss": ns.get("miss_reasons", [])[:4]
    } for ns in non_signals[:NEAR_MISS_LOG_COUNT]]

    for nm in near_misses:
        logging.info(f"Near-miss {nm['symbol']}: score={nm['score']} vz={nm['vz']} prox={nm['prox']} miss={nm['miss']}")

    payload: Dict[str, Any] = {
        "generated_at": human_time(started),
        "timezone": TZ_NAME,
        "regime": {"ok": regime_ok, "reason": regime_reason},
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
                "MIN_AVG_VOL": MIN_AVG_VOL,
                "STOP_ATR_BUFFER": STOP_ATR_BUFFER,
                "MIN_RISK_FLOOR_PCT": MIN_RISK_FLOOR_PCT,
                "MAX_RISK_CAP_PCT": MAX_RISK_CAP_PCT,
                "MAX_CANDIDATES": MAX_CANDIDATES,
                "RELAXED_SIGNALS": RELAXED_SIGNALS,
            },
            "lower_tf_confirm": True,
            "rs_reference": "BTCUSDC",
            "binance_endpoint": BASE_URL,
            "mode": mode,
            "near_misses": near_misses,
        }
    }
    return payload

def run_pipeline(mode: str, ignore_regime: bool, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
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

    # Backtest daily snapshots
    if start_date and end_date:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        day = start_dt; all_payloads = []
        while day <= end_dt:
            all_payloads.append(process(symbols, mode, ignore_regime, int(day.timestamp() * 1000)))
            day += timedelta(days=1)
        stats = {
            "total_days": len(all_payloads),
            "total_breakouts": sum(len(p.get("orders", [])) for p in all_payloads),
            "avg_candidates": (sum(len(p.get("candidates", [])) for p in all_payloads) / max(1, len(all_payloads))),
        }
        return {"backtest_results": all_payloads, "stats": stats}

    return process(symbols, mode, ignore_regime, None)

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyses pipeline")
    parser.add_argument("--mode", default="deep", choices=["deep", "light-fast", "light-hourly"])
    parser.add_argument("--ignore_regime", action="store_true", help="Ignore BTC regime check")
    parser.add_argument("--backtest", action="store_true", help="Run backtest over daily snapshots")
    parser.add_argument("--start_date", type=str)
    parser.add_argument("--end_date", type=str)
    args = parser.parse_args()

    try:
        payload = run_pipeline(args.mode, args.ignore_regime, args.start_date if args.backtest else None, args.end_date if args.backtest else None)
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
            "meta": {"binance_endpoint": BASE_URL, "mode": "error"},
        }

    latest_path = Path("public_runs/latest/summary.json")
    write_summary(payload, latest_path)

    stamp = datetime.now(timezone.utc).astimezone(TZ).strftime("%Y%m%d_%H%M%S")
    snap = Path("public_runs") / stamp / "summary.json"
    ensure_dirs(snap)
    with snap.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

    logging.info(f"Summary written to {latest_path} (signal={payload.get('signals',{}).get('type')}, regime_ok={payload.get('regime',{}).get('ok')})")

if __name__ == "__main__":
    main()