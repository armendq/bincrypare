#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses pipeline - scanner for pre-breakouts and breakouts (USDC universe)

- Universe: Binance Spot pairs with quoteAsset == USDC (TRADING, non-stable, non-leveraged)
- BTC reference: BTCUSDC
- Signals:
    B  = confirmed breakout
    C  = pre-breakout / continuation
- Output: public_runs/latest/summary.json and timestamped under public_runs/YYYYmmdd_HHMMSS/
- Notes:
    * 'missing_binance' is always [] to keep downstream contract stable.
    * Environment variables control thresholds; RELAXED_SIGNALS widens gates.
    * Optional REQUIRE_LOWER_TF_OK enforces 15m+5m momentum confirmation.
    * Feasibility check uses exchange filters so we do not emit untradeable ideas.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
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

# Core params (env-overridable)
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "200"))
SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "20"))

# Heuristics / thresholds (env)
PROX_ATR_MIN = float(os.getenv("PROX_ATR_MIN", "0.00"))
PROX_ATR_MAX = float(os.getenv("PROX_ATR_MAX", "0.60"))
VOL_Z_MIN_PRE = float(os.getenv("VOL_Z_MIN_PRE", "0.8"))
VOL_Z_MIN_BREAK = float(os.getenv("VOL_Z_MIN_BREAK", "1.0"))
BREAK_BUFFER_ATR = float(os.getenv("BREAK_BUFFER_ATR", "0.03"))
RELAX_B_HIGH = os.getenv("RELAX_B_HIGH", "0") == "1"
ALLOW_RUNAWAY_ATR = float(os.getenv("ALLOW_RUNAWAY_ATR", "3.0"))

# Liquidity / sizing metadata (advisory only)
MIN_AVG_VOL = float(os.getenv("MIN_AVG_VOL", "1500"))  # approx USDC across last 10 x 1h bars
CAPITAL = float(os.getenv("CAPITAL", "10000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))

# Relaxations and confirmations
RELAXED_SIGNALS = os.getenv("RELAXED_SIGNALS", "1") == "1"
REQUIRE_LOWER_TF_OK = os.getenv("REQUIRE_LOWER_TF_OK", "1") == "1"

# Feasibility pre-check to avoid suggesting untradeables on small accounts
ANALYSES_MIN_ORDER_USD = float(os.getenv("MIN_ORDER_USD", os.getenv("ANALYSES_MIN_ORDER_USD", "6")))

# Execution / logging
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "10"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "12"))
NEAR_MISS_LOG_COUNT = int(os.getenv("NEAR_MISS_LOG_COUNT", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="[%(asctime)s] %(message)s")

# Filters
STABLES = {"USDT", "USDC", "DAI", "TUSD", "USDP", "BUSD", "FDUSD", "PYUSD"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S")

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
    if series.size == 0 or period <= 1:
        return series
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, series.size):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out

def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = ATR_LEN) -> np.ndarray:
    if highs.size < 2:
        return np.zeros_like(highs)
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]))
    tr = np.maximum(tr, np.abs(lows[1:] - closes[:-1]))
    tr = np.insert(tr, 0, 0.0)  # align length
    return ema(tr, period)

def swing_high(series: np.ndarray, lookback: int) -> np.ndarray:
    return np.array([np.max(series[max(0, i - lookback + 1): i + 1]) for i in range(series.size)])

def swing_low(series: np.ndarray, lookback: int) -> np.ndarray:
    return np.array([np.min(series[max(0, i - lookback + 1): i + 1]) for i in range(series.size)])

def pct_change(series: np.ndarray, n: int) -> float:
    if series.size < n + 1:
        return 0.0
    a, b = float(series[-n - 1]), float(series[-1])
    if a == 0.0:
        return 0.0
    return (b - a) / a

def vol_zscore(vols: np.ndarray, window: int = 50) -> float:
    if vols.size < window + 1:
        return 0.0
    sample = vols[-(window + 1):-1]
    mu = float(np.mean(sample))
    sd = float(np.std(sample, ddof=0))
    if sd == 0.0:
        return 0.0
    return float((vols[-1] - mu) / sd)

# -------------------------- DATA FETCH --------------------------------

def _filters_from_symbol_meta(meta: dict) -> Dict[str, float]:
    """Extract useful filters (minQty, stepSize, tickSize, minNotional)."""
    fmin_qty = fstep = ftick = fmin_notional = 0.0
    for f in meta.get("filters", []):
        ft = f.get("filterType")
        if ft == "LOT_SIZE":
            fmin_qty = float(f.get("minQty", "0"))
            fstep = float(f.get("stepSize", "0"))
        elif ft == "PRICE_FILTER":
            ftick = float(f.get("tickSize", "0"))
        elif ft in ("NOTIONAL", "MIN_NOTIONAL"):
            try:
                fmin_notional = float(f.get("minNotional", "0"))
            except Exception:
                fmin_notional = 0.0
    return {
        "min_qty": fmin_qty,
        "step_qty": fstep,
        "tick": ftick,
        "min_notional": max(ANALYSES_MIN_ORDER_USD, fmin_notional or 0.0),
    }

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
        if isinstance(data, list):
            return data
    except Exception:
        return None
    return None

def fetch_all_klines(symbols: List[str], interval: str, limit: int, end_time: Optional[int] = None) -> Dict[str, Optional[List[List[Any]]]]:
    results: Dict[str, Optional[List[List[Any]]]] = {}
    if not symbols:
        return results
    with ThreadPoolExecutor(max_workers=max(1, min(MAX_WORKERS, len(symbols)))) as ex:
        fut = {ex.submit(fetch_klines, sym, interval, limit, end_time): sym for sym in symbols}
        for f in as_completed(fut):
            sym = fut[f]
            try:
                results[sym] = f.result()
            except Exception as e:
                logging.warning(f"Failed klines for {sym}: {e}")
                results[sym] = None
    return results

def parse_klines(raw: List[List[Any]]) -> Dict[str, np.ndarray]:
    return {
        "open_time": np.array([int(r[0]) for r in raw]),
        "open":      np.array([float(r[1]) for r in raw]),
        "high":      np.array([float(r[2]) for r in raw]),
        "low":       np.array([float(r[3]) for r in raw]),
        "close":     np.array([float(r[4]) for r in raw]),
        "volume":    np.array([float(r[5]) for r in raw]),
    }

# -------------------------- CORE LOGIC --------------------------------

def analyze_symbol(symbol: str, btc_1h_close: np.ndarray, req: Dict[str, Dict[str, np.ndarray]]) -> Optional[dict]:
    try:
        one = req["1h"]
        four = req["4h"]
        fifteen = req["15m"]
        five = req["5m"]

        close_1h = one["close"]
        high_1h  = one["high"]
        low_1h   = one["low"]
        vol_1h   = one["volume"]

        need_len = max(EMA_SLOW + 5, SWING_LOOKBACK + 5)
        if close_1h.size < need_len or four["close"].size < need_len:
            return None

        # Trends
        ema20_1h = ema(close_1h, EMA_FAST)
        ema200_1h = ema(close_1h, EMA_SLOW)
        ema20_4h = ema(four["close"], EMA_FAST)
        ema200_4h = ema(four["close"], EMA_SLOW)

        ema20_slope_1h = float(ema20_1h[-1] - ema20_1h[-4])
        ema20_slope_4h = float(ema20_4h[-1] - ema20_4h[-4])

        trend_ok = (
            ema20_slope_1h > 0 and
            ema20_slope_4h > 0 and
            float(close_1h[-1]) > float(ema200_1h[-1]) and
            float(four["close"][-1]) > float(ema200_4h[-1])
        )

        # ATR & swings (1h)
        atr_1h = atr(high_1h, low_1h, close_1h, ATR_LEN)
        if float(atr_1h[-1]) <= 0.0:
            return None

        hh20 = swing_high(high_1h, SWING_LOOKBACK)
        ll20 = swing_low(low_1h, SWING_LOOKBACK)

        # Volume and liquidity
        vz = vol_zscore(vol_1h, window=50)
        avg_vol_usdc = float(np.mean(vol_1h[-10:]) * close_1h[-1]) if vol_1h.size >= 10 else 0.0
        if avg_vol_usdc < MIN_AVG_VOL:
            return None

        # Proximity (in ATRs) - negative means already above HH20
        prox = float((hh20[-1] - close_1h[-1]) / max(1e-9, atr_1h[-1]))
        atr_rising = bool(atr_1h[-1] > atr_1h[-2] > atr_1h[-3])

        # Lower TF momentum confirms
        def mom_ok(tf: Dict[str, np.ndarray]) -> bool:
            c = tf["close"]
            e20 = ema(c, EMA_FAST)
            if e20.size < 5:
                return False
            return bool((c[-1] > e20[-1]) and (e20[-1] > e20[-3]))

        mom15 = mom_ok(fifteen)
        mom5  = mom_ok(five)
        lower_tf_ok = mom15 and mom5

        # RS vs BTC (1h)
        rs_strength = 0.0
        if btc_1h_close.size == close_1h.size:
            rs_series = close_1h / np.where(btc_1h_close != 0, btc_1h_close, 1e-9)
            rs_strength = pct_change(rs_series, n=10)

        last_close = float(close_1h[-1])
        last_high  = float(high_1h[-1])
        last_atr   = float(atr_1h[-1])
        breakout_level = float(hh20[-1])
        confirm_price = last_high if RELAX_B_HIGH else last_close

        # Relaxed gates
        trend_ok_relaxed = trend_ok or (RELAXED_SIGNALS and (lower_tf_ok or vz >= 0.5 or rs_strength > 0.1))

        # Optional hard requirement for lower TF
        if REQUIRE_LOWER_TF_OK and not lower_tf_ok:
            # Allow only if both trends strong AND vz very high, to not kill all signals in rare moments
            if not (trend_ok and vz >= max(VOL_Z_MIN_PRE, 1.2)):
                # record a miss reason by returning minimal info
                out = {"signal": "N", "miss_reasons": ["no_lowerTF_hard"]}
                return out  # upstream treats None as no-data; we want a counted analysis

        # Apply dynamic relax multipliers if RELAXED_SIGNALS
        vz_pre  = VOL_Z_MIN_PRE * (0.85 if RELAXED_SIGNALS else 1.0)
        vz_brk  = VOL_Z_MIN_BREAK * (0.8 if RELAXED_SIGNALS else 1.0)
        prox_min = PROX_ATR_MIN - (0.2 if RELAXED_SIGNALS else 0.0)  # allow slightly more above HH20
        prox_max = PROX_ATR_MAX + (0.3 if RELAXED_SIGNALS else 0.0)

        within_runaway = (last_close <= breakout_level + ALLOW_RUNAWAY_ATR * last_atr)

        # Confirmed breakout
        confirmed_breakout = (
            trend_ok_relaxed and
            (confirm_price > (breakout_level + BREAK_BUFFER_ATR * last_atr)) and
            (vz >= vz_brk) and
            (lower_tf_ok or RELAXED_SIGNALS) and
            within_runaway
        )

        # Pre-breakout / continuation
        pre_breakout = (
            trend_ok_relaxed and
            (prox_min <= prox <= prox_max) and
            (vz >= vz_pre) and
            (atr_rising or RELAXED_SIGNALS)
        )

        # Levels and advisory size
        entry = breakout_level + BREAK_BUFFER_ATR * last_atr
        stop  = float(ll20[-1]) - 0.3 * last_atr  # buffer under LL20
        t1    = entry + 0.8 * last_atr
        t2    = entry + 1.5 * last_atr

        entry_to_stop = max(entry - stop, 1e-9)
        position_size_usdc = (CAPITAL * RISK_PER_TRADE) / max(entry_to_stop / max(entry, 1e-9), 1e-9)
        position_size = position_size_usdc / max(entry, 1e-9)

        # Score (rankers)
        reasons: List[str] = []
        if trend_ok: reasons.append("TrendOK(1h&4h)")
        if atr_rising: reasons.append("ATR up")
        if vz > 0: reasons.append(f"VolZ={vz:.2f}")
        if lower_tf_ok: reasons.append("LowerTF OK")
        if rs_strength > 0: reasons.append(f"RS={rs_strength:.2f}")

        score = 0.0
        if vz > 0: score += vz
        if prox > 0: score += 1.0 / (min(max(prox, 0.01), 2.0))
        score += max(0.0, 5.0 * rs_strength)
        if lower_tf_ok: score += 0.5
        score += float(np.log1p(max(0.0, float(np.nan_to_num(np.array([avg_vol_usdc]))[0])) / 1e5))
        if not trend_ok and trend_ok_relaxed: score += 0.2

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
            "trend_ok": bool(trend_ok),
            "lower_tf_ok": bool(lower_tf_ok),
            "rs10": rs_strength,
            "score": score,
            "reasons": reasons,
            "avg_vol": avg_vol_usdc,
            "position_size": position_size,
            "position_size_usdc": position_size_usdc,
        }

        if confirmed_breakout:
            out["signal"] = "B"
        elif pre_breakout:
            out["signal"] = "C"
        else:
            out["signal"] = "N"
            miss = []
            if not trend_ok_relaxed: miss.append("trend")
            if vz < min(vz_pre, vz_brk): miss.append(f"low_vz={vz:.2f}")
            if not (prox_min <= prox <= prox_max): miss.append(f"prox={prox:.2f}")
            if not lower_tf_ok: miss.append("no_lowerTF")
            if not within_runaway: miss.append("too_far_runaway")
            out["miss_reasons"] = miss

        return out

    except Exception as e:
        logging.warning(f"Analysis failed for {symbol}: {e}")
        return None

def write_summary(payload: dict, dest_latest: Path) -> None:
    ensure_dirs(dest_latest)
    with dest_latest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

# -------------------------- PIPELINE ----------------------------------

def build_universe() -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    """
    Returns:
      mapping: {base: symbol}
      filt: {symbol: {min_qty, step_qty, tick, min_notional}}
    """
    exch = fetch_exchange_usdc_symbols()
    if not exch:
        return {}, {}
    mapping: Dict[str, str] = {}
    filt: Dict[str, Dict[str, float]] = {}
    for sym, meta in exch.items():
        base = str(meta.get("baseAsset", "")).upper()
        if not base:
            continue
        mapping[base] = sym
        filt[sym] = _filters_from_symbol_meta(meta)
    return mapping, filt

def compute_regime(btc_1h: Dict[str, np.ndarray], btc_4h: Dict[str, np.ndarray]) -> Tuple[bool, str]:
    try:
        ema200_4h = ema(btc_4h["close"], EMA_SLOW)
        ema20_4h = ema(btc_4h["close"], EMA_FAST)
        ema200_1h = ema(btc_1h["close"], EMA_SLOW)
        ema20_1h = ema(btc_1h["close"], EMA_FAST)
        if ema200_4h.size > 4 and ema200_1h.size > 4:
            slope4 = float(ema20_4h[-1] - ema20_4h[-4])
            slope1 = float(ema20_1h[-1] - ema20_1h[-4])
            if (btc_4h["close"][-1] > ema200_4h[-1]) and (btc_1h["close"][-1] > ema200_1h[-1]) and slope4 > 0 and slope1 > 0:
                return True, "BTC uptrend (4h & 1h)"
            return False, "BTC not in uptrend"
        return False, "insufficient data"
    except Exception:
        return False, "regime calc error"

def process(symbols: List[str], sym_filters: Dict[str, Dict[str, float]],
            mode: str, ignore_regime: bool, end_time: Optional[int] = None) -> dict:
    started = datetime.now(timezone.utc).astimezone(TZ) if end_time is None else datetime.fromtimestamp(end_time / 1000, timezone.utc).astimezone(TZ)

    # Baseline BTC
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

    btc_1h = parse_klines(btc_raw_1h)
    btc_4h = parse_klines(btc_raw_4h)
    regime_ok, regime_reason = compute_regime(btc_1h, btc_4h)
    if ignore_regime:
        regime_ok = True
        regime_reason = "Regime ignored by flag"
    btc_close_1h = btc_1h["close"]

    # Thin symbols in light modes
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
    no_data: List[str] = []
    all_infos: List[dict] = []
    scanned = 0
    near_miss_compact: List[dict] = []

    for sym in symbols:
        scanned += 1
        tf_data = {tf: reqs[tf][sym] for tf in INTERVALS}
        if any(v is None for v in tf_data.values()):
            no_data.append(sym.split("USDC")[0])
            continue

        info = analyze_symbol(sym, btc_close_1h, tf_data)  # type: ignore
        if not info:
            no_data.append(sym.split("USDC")[0])
            continue

        # Ensure tkr and rotation flag
        info["ticker"] = sym.split("USDC")[0]
        info["rotation_exempt"] = False

        # Feasibility pre-check vs filters (avoid untradeables)
        f = sym_filters.get(sym, {"min_qty": 0.0, "step_qty": 0.0, "tick": 0.0, "min_notional": ANALYSES_MIN_ORDER_USD})
        entry = float(info.get("entry", 0.0))
        min_notional = float(f.get("min_notional", ANALYSES_MIN_ORDER_USD)) or ANALYSES_MIN_ORDER_USD
        min_qty = float(f.get("min_qty", 0.0))
        feasible = (entry > 0) and ((entry * max(min_qty, ANALYSES_MIN_ORDER_USD / max(entry, 1e-12))) >= min_notional - 1e-9)
        info["feasible"] = bool(feasible)
        info["filters"] = {"min_qty": f["min_qty"], "step_qty": f["step_qty"], "tick": f["tick"], "min_notional": min_notional}

        # Sort into buckets
        all_infos.append(info)
        sig = info.get("signal")
        if sig == "B":
            if feasible:
                confirmed.append(info)
        elif sig == "C":
            if feasible:
                candidates.append(info)
        else:
            # compact near-miss collection for payload
            if len(near_miss_compact) < NEAR_MISS_LOG_COUNT:
                near_miss_compact.append({
                    "symbol": sym,
                    "vz": round(float(info.get("vol_z", 0.0)), 2),
                    "prox": round(float(info.get("prox_atr", 0.0)), 2),
                    "score": round(float(info.get("score", 0.0)), 2),
                    "miss": info.get("miss_reasons", []),
                })

    # Near-miss logging (top N non-signals)
    non_signals = [i for i in all_infos if i.get("signal") == "N"]
    non_signals.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    for ns in non_signals[:NEAR_MISS_LOG_COUNT]:
        logging.info(f"Near-miss {ns['symbol']}: score={ns.get('score',0):.2f}, vz={ns.get('vol_z',0):.2f}, prox={ns.get('prox_atr',0):.2f}, miss={ns.get('miss_reasons', [])}")

    if not regime_ok:
        return {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": regime_ok, "reason": regime_reason},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": scanned, "eligible": len(symbols), "skipped": {"no_data": sorted(no_data), "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode, "near_misses": near_miss_compact},
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
                "filters": o.get("filters", {}),
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
        "rs10": round(float(c.get("rs10", 0.0)), 4),
        "tf": "1h",
        "notes": c.get("reasons", []),
        "rotation_exempt": False,
        "position_size": round(float(c["position_size"]), 8),
        "position_size_usdc": round(float(c["position_size_usdc"]), 2),
        "filters": c.get("filters", {}),
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
            "eligible": len(symbols),
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
                "ALLOW_RUNAWAY_ATR": ALLOW_RUNAWAY_ATR,
                "MAX_CANDIDATES": MAX_CANDIDATES,
                "MIN_AVG_VOL": MIN_AVG_VOL,
                "CAPITAL": CAPITAL,
                "RISK_PER_TRADE": RISK_PER_TRADE,
                "RELAXED_SIGNALS": RELAXED_SIGNALS,
                "REQUIRE_LOWER_TF_OK": REQUIRE_LOWER_TF_OK,
                "ANALYSES_MIN_ORDER_USD": ANALYSES_MIN_ORDER_USD,
            },
            "lower_tf_confirm": True,
            "rs_reference": "BTCUSDC",
            "binance_endpoint": BASE_URL,
            "mode": mode,
            "near_misses": near_miss_compact,
        }
    }
    return payload

def run_pipeline(mode: str, ignore_regime: bool = False,
                 start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
    # Backtest mode (optional)
    if start_date and end_date:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        cur = start_dt
        results = []
        while cur <= end_dt:
            end_ms = int(cur.timestamp() * 1000)
            mapping, filt = build_universe()
            symbols = list(mapping.values())
            results.append(process(symbols, filt, mode, ignore_regime, end_ms))
            cur += timedelta(days=1)
        stats = {
            "total_days": len(results),
            "total_breakouts": sum(len(p.get("orders", [])) for p in results),
            "avg_candidates": (sum(len(p.get("candidates", [])) for p in results) / max(1, len(results))),
        }
        return {"backtest_results": results, "stats": stats}

    # Live mode
    mapping, filt = build_universe()
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
    return process(symbols, filt, mode, ignore_regime)

# ------------------------------ CLI -----------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyses pipeline (USDC)")
    parser.add_argument("--mode", default="deep", choices=["deep", "light-fast", "light-hourly"])
    parser.add_argument("--ignore_regime", action="store_true", help="Ignore BTC regime check")
    parser.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    parser.add_argument("--start_date", type=str, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date for backtest (YYYY-MM-DD)")
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

    logging.info(f"Summary written to {latest_path} "
                 f"(signal={payload.get('signals',{}).get('type')}, regime_ok={payload.get('regime',{}).get('ok')})")

if __name__ == "__main__":
    main()