#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses pipeline - scanner for pre-breakouts and breakouts (USDC universe)

- Universe: all Binance Spot pairs with quoteAsset == USDC
- Excludes bases that are stablecoins and leveraged/multiplier tokens
- BTC reference: BTCUSDC
- Emits public_runs/latest/summary.json and timestamped copy
- 'missing_binance' is always empty (contract stability)
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

TZ_NAME = "Europe/Prague"
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
ATR_LEN = 14
EMA_FAST = 20
EMA_SLOW = 200
SWING_LOOKBACK = 20  # HH20 / LL20

# Relaxed thresholds (env-overridable)
PROX_ATR_MIN     = float(os.getenv("PROX_ATR_MIN", "0.02"))
PROX_ATR_MAX     = float(os.getenv("PROX_ATR_MAX", "0.50"))
VOL_Z_MIN_PRE    = float(os.getenv("VOL_Z_MIN_PRE", "1.0"))
VOL_Z_MIN_BREAK  = float(os.getenv("VOL_Z_MIN_BREAK", "1.2"))
BREAK_BUFFER_ATR = float(os.getenv("BREAK_BUFFER_ATR", "0.03"))
RELAX_B_HIGH     = os.getenv("RELAX_B_HIGH", "0") == "1"   # 0=use close, 1=use high

# Liquidity & risk (advisory sizing)
MIN_AVG_VOL_USDC   = float(os.getenv("MIN_AVG_VOL", "4000"))
CAPITAL            = float(os.getenv("CAPITAL", "10000"))
RISK_PER_TRADE     = float(os.getenv("RISK_PER_TRADE", "0.01"))
MIN_RISK_FLOOR_PCT = float(os.getenv("MIN_RISK_FLOOR_PCT", "0.01"))
MAX_RISK_CAP_PCT   = float(os.getenv("MAX_RISK_CAP_PCT", "0.04"))
STOP_ATR_BUFFER    = float(os.getenv("STOP_ATR_BUFFER", "0.4"))  # LL20 - 0.4*ATR

# Ranking
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "10"))

# Filters
STABLES = {"USDT", "USDC", "DAI", "TUSD", "USDP", "BUSD", "FDUSD", "PYUSD"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S")

# Concurrency
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="[%(asctime)s] %(levelname)s %(message)s")


# -------------------------- HELPERS -----------------------------------

def human_time(ts: Optional[datetime] = None) -> str:
    dt = ts or datetime.now(timezone.utc).astimezone(TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def safe_get(url: str, params: Optional[dict] = None, retries: int = 5, timeout: int = 20, sleep_s: float = 0.8) -> Optional[requests.Response]:
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
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out

def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = ATR_LEN) -> np.ndarray:
    if len(highs) < 2:
        return np.zeros_like(highs, dtype=np.float64)
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    trs = np.maximum(tr1, np.maximum(tr2, tr3))  # len N-1
    trs = np.concatenate(([0.0], trs))           # prepend 0 to align
    return ema(trs, period)

def swing_high(series: np.ndarray, lookback: int) -> np.ndarray:
    return np.array([np.max(series[max(0, i - lookback + 1): i + 1]) for i in range(len(series))], dtype=np.float64)

def swing_low(series: np.ndarray, lookback: int) -> np.ndarray:
    return np.array([np.min(series[max(0, i - lookback + 1): i + 1]) for i in range(len(series))], dtype=np.float64)

def pct_change(series: np.ndarray, n: int) -> float:
    if len(series) < n + 1:
        return 0.0
    a, b = series[-n - 1], series[-1]
    if a == 0:
        return 0.0
    return float((b - a) / a)

def vol_zscore(vols: np.ndarray, window: int = 50) -> float:
    if len(vols) < window + 1:
        return 0.0
    sample = vols[-(window + 1):-1]
    mu = float(np.mean(sample))
    sd = float(np.std(sample, ddof=0))
    if sd == 0:
        return 0.0
    return float((vols[-1] - mu) / sd)

def fetch_exchange_usdc_symbols() -> Dict[str, dict]:
    url = f"{BASE_URL}/api/v3/exchangeInfo"
    r = safe_get(url, retries=5, timeout=25)
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
    if end_time:
        params["endTime"] = end_time
    r = safe_get(url, params=params, retries=5, timeout=20)
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
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_klines, sym, interval, limit, end_time): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                results[sym] = future.result()
            except Exception as e:
                logging.warning(f"Failed to fetch klines for {sym}: {e}")
                time.sleep(0.2)
                results[sym] = None
    return results

def refetch_missing_tf(tf_map: Dict[str, Optional[List[List[Any]]]], symbols: List[str], interval: str, limit: int, end_time: Optional[int]) -> Dict[str, Optional[List[List[Any]]]]:
    missing = [s for s, v in tf_map.items() if v is None]
    if not missing:
        return tf_map
    time.sleep(0.6)  # backoff
    for s in missing:
        tf_map[s] = fetch_klines(s, interval, limit, end_time)
        time.sleep(0.05)
    return tf_map

def parse_klines(raw: List[List[Any]]) -> Dict[str, np.ndarray]:
    to_f = lambda arr: np.array(arr, dtype=np.float64)
    return {
        "open_time": np.array([int(r[0]) for r in raw], dtype=np.int64),
        "open":  to_f([r[1] for r in raw]),
        "high":  to_f([r[2] for r in raw]),
        "low":   to_f([r[3] for r in raw]),
        "close": to_f([r[4] for r in raw]),
        "volume":to_f([r[5] for r in raw]),
    }

# -------------------------- CORE LOGIC --------------------------------

def analyze_symbol(symbol: str, btc_1h_close: np.ndarray, req: Dict[str, Dict[str, np.ndarray]]) -> Optional[dict]:
    try:
        one = req["1h"]; four = req["4h"]; fifteen = req["15m"]; five = req["5m"]

        close_1h = one["close"]; high_1h = one["high"]; low_1h = one["low"]; vol_1h = one["volume"]
        if len(close_1h) < max(EMA_SLOW + 5, SWING_LOOKBACK + 5):
            return None

        # Trend filters
        ema20_1h = ema(close_1h, EMA_FAST)
        ema200_1h = ema(close_1h, EMA_SLOW)
        ema20_4h = ema(four["close"], EMA_FAST)
        ema200_4h = ema(four["close"], EMA_SLOW)
        if len(ema200_1h) < 5 or len(ema200_4h) < 5:
            return None

        ema20_slope_1h = float(ema20_1h[-1] - ema20_1h[-4])
        ema20_slope_4h = float(ema20_4h[-1] - ema20_4h[-4])
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

        # Liquidity (robust trimmed mean)
        window = 12 if len(vol_1h) >= 12 else len(vol_1h)
        last = vol_1h[-window:]
        if len(last) >= 6:
            q1, q9 = np.quantile(last, [0.15, 0.85])
            trim = last[(last >= q1) & (last <= q9)]
        else:
            trim = last
        avg_vol_usdc = float((np.mean(trim) if len(trim) else np.mean(last)) * close_1h[-1])
        if avg_vol_usdc < MIN_AVG_VOL_USDC:
            return None

        # Pre-breakout proximity
        prox = float((hh20[-1] - close_1h[-1]) / max(1e-9, atr_1h[-1]))
        atr_rising = bool(atr_1h[-1] > atr_1h[-2] > atr_1h[-3])

        # Lower TF momentum
        def mom_ok(tf_data: Dict[str, np.ndarray]) -> bool:
            c = tf_data["close"]
            e20 = ema(c, EMA_FAST)
            if len(e20) < 5:
                return False
            return (c[-1] > e20[-1]) and (e20[-1] > e20[-3])

        lower_tf_ok = mom_ok(fifteen) and mom_ok(five)

        # RS vs BTC (1h)
        rs_strength = 0.0
        if len(btc_1h_close) == len(close_1h):
            denom = np.where(btc_1h_close != 0, btc_1h_close, 1e-9)
            rs_series = close_1h / denom
            rs_strength = pct_change(rs_series, n=10)

        last_close = float(close_1h[-1])
        last_high  = float(high_1h[-1])
        last_atr   = float(atr_1h[-1])
        breakout_level = float(hh20[-1])

        reasons: List[str] = []
        if trend_ok: reasons.append("TrendOK")
        if atr_rising: reasons.append("ATR up")
        if vz > 0: reasons.append(f"VolZ={vz:.2f}")
        if lower_tf_ok: reasons.append("LowerTF OK")

        # Score
        score = 0.0
        if vz > 0: score += vz
        if prox > 0: score += 1.0 / (min(max(prox, 0.01), 2.0))
        score += max(0.0, 5.0 * rs_strength)
        if lower_tf_ok: score += 0.5
        score += float(np.log1p(avg_vol_usdc / 1e5))  # liquidity boost

        # Confirmation choice
        confirm_price = last_high if RELAX_B_HIGH else last_close
        confirmed_breakout = (
            trend_ok
            and confirm_price > (breakout_level + BREAK_BUFFER_ATR * last_atr)
            and vz >= VOL_Z_MIN_BREAK
            and lower_tf_ok
        )

        # Pre-breakout
        pre_breakout = (
            trend_ok
            and atr_rising
            and PROX_ATR_MIN <= prox <= PROX_ATR_MAX
            and vz >= VOL_Z_MIN_PRE
        )

        # Levels
        entry = breakout_level + BREAK_BUFFER_ATR * last_atr

        # Stop: LL20 - buffer*ATR, with min and max risk caps
        base_stop = float(ll20[-1]) - STOP_ATR_BUFFER * last_atr
        natural_risk = max(entry - base_stop, 0.0)
        min_risk_width = MIN_RISK_FLOOR_PCT * entry
        max_risk_width = MAX_RISK_CAP_PCT * entry
        entry_to_stop = min(max(natural_risk, max(min_risk_width, 1e-9)), max_risk_width)
        stop = entry - entry_to_stop

        t1 = entry + 0.8 * last_atr
        t2 = entry + 1.5 * last_atr

        # Advisory sizing (display only)
        position_size = 0.0
        position_size_usdc = 0.0
        if entry > 0 and entry_to_stop > 0:
            position_size_usdc = (CAPITAL * RISK_PER_TRADE) / (entry_to_stop / entry)
            position_size = position_size_usdc / entry

        out: Dict[str, Any] = {
            "symbol": symbol,
            "last": last_close,
            "atr": last_atr,
            "hh20": breakout_level,
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
        logging.warning(f"Analysis failed for {symbol}: {e}")
        return None

def write_summary(payload: dict, dest_latest: Path) -> None:
    ensure_dirs(dest_latest)
    try:
        with dest_latest.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
    except Exception as e:
        logging.error(f"Failed writing {dest_latest}: {e}")

# -------------------------- MAIN --------------------------------------

def build_universe() -> Tuple[Dict[str, str], List[str]]:
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

def compute_regime(btc_1h: Dict[str, np.ndarray], btc_4h: Dict[str, np.ndarray]) -> Tuple[bool, str]:
    try:
        ema200_4h = ema(btc_4h["close"], EMA_SLOW)
        ema20_4h  = ema(btc_4h["close"], EMA_FAST)
        ema200_1h = ema(btc_1h["close"], EMA_SLOW)
        ema20_1h  = ema(btc_1h["close"], EMA_FAST)
        if len(ema200_4h) > 4 and len(ema200_1h) > 4:
            slope4 = float(ema20_4h[-1] - ema20_4h[-4])
            slope1 = float(ema20_1h[-1] - ema20_1h[-4])
            if (btc_4h["close"][-1] > ema200_4h[-1]) and (btc_1h["close"][-1] > ema200_1h[-1]) and slope4 > 0 and slope1 > 0:
                return True, "BTC uptrend (4h & 1h)"
            return False, "BTC not in uptrend"
        return False, "insufficient data"
    except Exception:
        return False, "regime calc error"

def run_pipeline(mode: str, ignore_regime: bool = False, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
    mapping, _ = build_universe()
    symbols = list(mapping.values())

    if not symbols:
        return {
            "generated_at": human_time(),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "exchangeInfo unavailable or no eligible symbols"},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": 0, "eligible": 0, "skipped": {"no_data": [], "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode, "sizing_in_scanner": True},
        }

    # Backtest range
    if start_date and end_date:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt   = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        current_dt = start_dt
        all_payloads = []
        while current_dt <= end_dt:
            end_time_ms = int(current_dt.timestamp() * 1000)
            payload = process_batch(symbols, mode, ignore_regime, end_time_ms)
            all_payloads.append(payload)
            current_dt += timedelta(days=1)
        total_days = len(all_payloads)
        total_breakouts = sum(len(p.get("orders", [])) for p in all_payloads if isinstance(p, dict))
        total_candidates = sum(len(p.get("candidates", [])) for p in all_payloads if isinstance(p, dict))
        stats = {
            "total_days": total_days,
            "total_breakouts": total_breakouts,
            "avg_breakouts_per_day": (total_breakouts / total_days) if total_days else 0.0,
            "avg_candidates_per_day": (total_candidates / total_days) if total_days else 0.0,
        }
        return {"backtest_results": all_payloads, "stats": stats}

    return process_batch(symbols, mode, ignore_regime)

def process_batch(symbols: List[str], mode: str, ignore_regime: bool, end_time: Optional[int] = None) -> dict:
    started = datetime.now(timezone.utc).astimezone(TZ) if not end_time else datetime.fromtimestamp(end_time / 1000, timezone.utc).astimezone(TZ)

    # RS baseline: BTCUSDC
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
            "meta": {"binance_endpoint": BASE_URL, "mode": mode, "sizing_in_scanner": True},
        }

    btc_1h = parse_klines(btc_raw_1h)
    btc_4h = parse_klines(btc_raw_4h)
    regime_ok, regime_reason = compute_regime(btc_1h, btc_4h)
    if ignore_regime:
        regime_ok = True
        regime_reason = "Regime ignored by flag"
    btc_close_1h = btc_1h["close"]

    # Randomized thinning for light modes (seeded by minute)
    if mode in ("light-fast", "light-hourly"):
        seed = int(datetime.utcnow().strftime("%Y%m%d%H%M"))
        rng = random.Random(seed)
        rng.shuffle(symbols)
        frac = 0.5 if mode == "light-fast" else (1.0 / 3.0)
        symbols = symbols[: max(1, int(len(symbols) * frac))]

    # Parallel fetch all TFs with one retry pass for missing
    reqs: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for tf, (interval, limit) in INTERVALS.items():
        raw_all = fetch_all_klines(symbols, interval, limit, end_time)
        raw_all = refetch_missing_tf(raw_all, symbols, interval, limit, end_time)
        reqs[tf] = {sym: (parse_klines(raw_all[sym]) if raw_all[sym] else None) for sym in symbols}

    candidates: List[dict] = []
    confirmed:  List[dict] = []
    no_data:    List[str] = []
    scanned = 0
    reasons_breakdown = {"short_history": 0, "tf_missing": 0, "liquidity_or_filters": 0, "ok": 0}

    for sym in symbols:
        scanned += 1
        tf_data = {tf: reqs[tf][sym] for tf in INTERVALS}
        if any(d is None for d in tf_data.values()):
            no_data.append(sym.split("USDC")[0])
            reasons_breakdown["tf_missing"] += 1
            continue

        try:
            short_hist_flag = (
                len(tf_data["1h"]["close"]) < EMA_SLOW + 5
                or len(tf_data["4h"]["close"]) < EMA_SLOW + 5
            )
        except Exception:
            short_hist_flag = False

        info = analyze_symbol(sym, btc_close_1h, tf_data)
        if not info:
            no_data.append(sym.split("USDC")[0])
            if short_hist_flag:
                reasons_breakdown["short_history"] += 1
            else:
                reasons_breakdown["liquidity_or_filters"] += 1
            continue

        info["ticker"] = sym.split("USDC")[0]
        info["rotation_exempt"] = False
        reasons_breakdown["ok"] += 1
        if info["signal"] == "B":
            confirmed.append(info)
        elif info["signal"] == "C":
            candidates.append(info)

    logging.info(
        f"scanned={scanned} ok={reasons_breakdown['ok']} tf_missing={reasons_breakdown['tf_missing']} "
        f"short_hist={reasons_breakdown['short_history']} liquidity_or_filters={reasons_breakdown['liquidity_or_filters']}"
    )

    # If regime off, do not trade
    if not regime_ok:
        payload = {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": regime_ok, "reason": regime_reason},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": scanned, "eligible": len(symbols), "skipped": {"no_data": sorted(no_data), "missing_binance": []}},
            "meta": {
                "binance_endpoint": BASE_URL,
                "mode": mode,
                "sizing_in_scanner": True,
                "no_data_breakdown": reasons_breakdown,
            },
        }
        return payload

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
                "position_size": round(o["position_size"], 8),
                "position_size_usdc": round(o["position_size_usdc"], 2),
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
        "position_size": round(c["position_size"], 8),
        "position_size_usdc": round(c["position_size_usdc"], 2),
    } for c in top_candidates]

    # debug_top if nothing passed
    debug_top = []
    if not confirmed and not candidates:
        tmp = []
        for sym in symbols:
            tf_data = {tf: reqs[tf][sym] for tf in INTERVALS}
            if any(d is None for d in tf_data.values()):
                continue
            info = analyze_symbol(sym, btc_close_1h, tf_data)
            if info:
                tmp.append(info)
        tmp.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        debug_top = [{
            "symbol": x["symbol"],
            "last": round(x["last"], 8),
            "score": round(x["score"], 4),
            "vol_z": round(x["vol_z"], 2),
            "prox_atr": round(x["prox_atr"], 3),
            "trend_ok": x["trend_ok"],
            "lower_tf_ok": x["lower_tf_ok"],
            "entry": round(x["entry"], 8),
            "hh20": round(x["hh20"], 8),
        } for x in tmp[:10]]

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
                "MAX_CANDIDATES": MAX_CANDIDATES,
                "MIN_AVG_VOL_USDC": MIN_AVG_VOL_USDC,
                "CAPITAL": CAPTIAL if (CAPTIAL:=CAPITAL) else CAPITAL,  # keep value in payload
                "RISK_PER_TRADE": RISK_PER_TRADE,
                "MIN_RISK_FLOOR_PCT": MIN_RISK_FLOOR_PCT,
                "MAX_RISK_CAP_PCT": MAX_RISK_CAP_PCT,
                "STOP_ATR_BUFFER": STOP_ATR_BUFFER,
                "MAX_WORKERS": MAX_WORKERS,
            },
            "lower_tf_confirm": True,
            "rs_reference": "BTCUSDC",
            "binance_endpoint": BASE_URL,
            "mode": mode,
            "sizing_in_scanner": True,
            "no_data_breakdown": reasons_breakdown,
            "debug_top": debug_top,
        }
    }
    return payload

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyses pipeline")
    parser.add_argument("--mode", default="deep", choices=["deep", "light-fast", "light-hourly"])
    parser.add_argument("--ignore_regime", action="store_true", help="Ignore BTC regime check")
    parser.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    parser.add_argument("--start_date", type=str, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date for backtest (YYYY-MM-DD)")
    args = parser.parse_args()

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
            "meta": {"binance_endpoint": BASE_URL, "mode": args.mode, "sizing_in_scanner": True},
        }

    latest_path = Path("public_runs/latest/summary.json")
    write_summary(payload, latest_path)

    stamp = datetime.now(timezone.utc).astimezone(TZ).strftime("%Y%m%d_%H%M%S")
    snapshot_dir = Path("public_runs") / stamp
    ensure_dirs(snapshot_dir / "summary.json")
    try:
        with (snapshot_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
    except Exception as e:
        logging.error(f"Failed writing snapshot: {e}")

    logging.info(f"Summary written to {latest_path} (signal={payload.get('signals',{}).get('type')}, regime_ok={payload.get('regime',{}).get('ok')})")

if __name__ == "__main__":
    main()