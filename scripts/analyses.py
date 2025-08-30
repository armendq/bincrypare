#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses pipeline – USDC universe (breakouts & pre-breakouts).

- Universe: all Binance Spot pairs with quoteAsset == USDC (ex-stables, ex-leveraged)
- RS baseline: BTCUSDC (can be ignored with --ignore_regime)
- Parallel data fetch for 4h/1h/15m/5m
- Candidate ranking & JSON output to public_runs/latest/summary.json
- NEW: size feasibility check (minQty / stepSize / (MIN_)NOTIONAL) so we don't suggest
       trades that Binance will reject for too-small notional/precision.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

# ATR / EMA
ATR_LEN = 14
EMA_FAST = 20
EMA_SLOW = 200
SWING_LOOKBACK = 20

# Heuristics (env-overridable)
PROX_ATR_MIN = float(os.getenv("PROX_ATR_MIN", "0.05"))
PROX_ATR_MAX = float(os.getenv("PROX_ATR_MAX", "0.35"))
VOL_Z_MIN_PRE = float(os.getenv("VOL_Z_MIN_PRE", "1.0"))
VOL_Z_MIN_BREAK = float(os.getenv("VOL_Z_MIN_BREAK", "1.2"))
BREAK_BUFFER_ATR = float(os.getenv("BREAK_BUFFER_ATR", "0.05"))
RELAX_B_HIGH = os.getenv("RELAX_B_HIGH", "0") == "1"    # confirm with last 1h HIGH if true
RELAXED_SIGNALS = os.getenv("RELAXED_SIGNALS", "0") == "1"

# Liquidity / sizing assumptions for feasibility filter
MIN_AVG_VOL = float(os.getenv("MIN_AVG_VOL", "2500"))             # USDC over last 10 1h bars
CAPITAL = float(os.getenv("CAPITAL", "10000"))                    # used for advisory sizing
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))       # 1%
ANALYSES_EQUITY_USDC = float(os.getenv("ANALYSES_EQUITY_USDC", "30"))
MIN_NOTIONAL_FALLBACK = float(os.getenv("MIN_NOTIONAL_FALLBACK", "5.0"))

# Score / logging
MAX_CANDIDATES = 10
NEAR_MISS_LOG_COUNT = int(os.getenv("NEAR_MISS_LOG_COUNT", "5"))

# Filters
STABLES = {"USDT", "USDC", "DAI", "TUSD", "USDP", "BUSD", "FDUSD", "PYUSD"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S")

logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
                    format="[%(asctime)s] %(message)s")

# -------------------------- HTTP / UTILS ------------------------------

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

# -------------------------- TA HELPERS --------------------------------

def ema(series: np.ndarray, period: int) -> np.ndarray:
    if len(series) == 0 or period <= 1:
        return series
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i-1]
    return out

def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = ATR_LEN) -> np.ndarray:
    if len(highs) < 2:
        return np.zeros_like(highs)
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    trs = np.insert(np.maximum(tr1, np.maximum(tr2, tr3)), 0, 0.0)
    return ema(trs, period)

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
    sd = float(np.std(sample, ddof=0))
    if sd == 0:
        return 0.0
    return (float(vols[-1]) - mu) / sd

# ---------------------- UNIVERSE + FILTERS ----------------------------

_SYMBOL_FILTERS: Dict[str, Dict[str, float]] = {}

def fetch_exchange_usdc_symbols() -> Dict[str, dict]:
    """exchangeInfo filtered to USDC spot symbols (ex-stable, ex-leveraged)."""
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

def load_symbol_filters() -> Dict[str, Dict[str, float]]:
    """Build {symbol: {min_qty, step_qty, tick, min_notional}} once."""
    global _SYMBOL_FILTERS
    if _SYMBOL_FILTERS:
        return _SYMBOL_FILTERS
    ex = fetch_exchange_usdc_symbols()
    filt: Dict[str, Dict[str, float]] = {}
    for sym, meta in ex.items():
        min_qty = step_qty = tick = 0.0
        min_notional = MIN_NOTIONAL_FALLBACK
        for f in meta.get("filters", []):
            t = f.get("filterType")
            if t == "LOT_SIZE":
                min_qty = float(f.get("minQty", "0"))
                step_qty = float(f.get("stepSize", "0"))
            elif t == "PRICE_FILTER":
                tick = float(f.get("tickSize", "0"))
            elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                try:
                    min_notional = float(f.get("minNotional", MIN_NOTIONAL_FALLBACK))
                except Exception:
                    min_notional = MIN_NOTIONAL_FALLBACK
        filt[sym] = {
            "min_qty": max(0.0, min_qty),
            "step_qty": max(0.0, step_qty),
            "tick": max(0.0, tick),
            "min_notional": max(MIN_NOTIONAL_FALLBACK, min_notional),
        }
    _SYMBOL_FILTERS = filt
    return filt

def fetch_klines(symbol: str, interval: str, limit: int, end_time: Optional[int] = None) -> Optional[List[List[Any]]]:
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
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
    with ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", "12"))) as ex:
        futures = {ex.submit(fetch_klines, s, interval, limit, end_time): s for s in symbols}
        res: Dict[str, Optional[List[List[Any]]]] = {}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                res[sym] = fut.result()
            except Exception as e:
                logging.warning(f"Failed klines {sym}: {e}")
                res[sym] = None
        return res

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

def quantize_qty(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step

def analyze_symbol(symbol: str, btc_1h_close: np.ndarray, req: Dict[str, Dict[str, np.ndarray]]) -> Optional[dict]:
    try:
        one = req["1h"]; four = req["4h"]; fifteen = req["15m"]; five = req["5m"]

        close_1h = one["close"]; high_1h = one["high"]; low_1h = one["low"]; vol_1h = one["volume"]
        if len(close_1h) < max(EMA_SLOW + 5, SWING_LOOKBACK + 5):
            return None

        # Trend
        ema20_1h = ema(close_1h, EMA_FAST); ema200_1h = ema(close_1h, EMA_SLOW)
        ema20_4h = ema(four["close"], EMA_FAST); ema200_4h = ema(four["close"], EMA_SLOW)
        ema20_slope_1h = float(ema20_1h[-1] - ema20_1h[-4])
        ema20_slope_4h = float(ema20_4h[-1] - ema20_4h[-4])
        trend_ok = (ema20_slope_1h > 0 and ema20_slope_4h > 0 and
                    float(close_1h[-1]) > float(ema200_1h[-1]) and float(four["close"][-1]) > float(ema200_4h[-1]))

        # ATR/swing
        atr_1h = atr(high_1h, low_1h, close_1h, ATR_LEN)
        if float(atr_1h[-1]) <= 0:
            return None
        hh20 = swing_high(high_1h, SWING_LOOKBACK)
        ll20 = swing_low(low_1h, SWING_LOOKBACK)

        # Volume / Liquidity
        vz = vol_zscore(vol_1h, window=50)
        avg_vol_usdc = float(np.mean(vol_1h[-10:]) * close_1h[-1])
        if avg_vol_usdc < MIN_AVG_VOL:
            return None

        # Proximity & momentum confirms
        prox = (float(hh20[-1]) - float(close_1h[-1])) / max(1e-9, float(atr_1h[-1]))
        atr_rising = float(atr_1h[-1]) > float(atr_1h[-2]) > float(atr_1h[-3])

        def mom_ok(tf: Dict[str, np.ndarray]) -> bool:
            c = tf["close"]; e20 = ema(c, EMA_FAST)
            return len(e20) >= 5 and (float(c[-1]) > float(e20[-1])) and (float(e20[-1]) > float(e20[-3]))
        lower_tf_ok = mom_ok(fifteen) and mom_ok(five)

        # RS vs BTC
        rs_strength = 0.0
        if len(btc_1h_close) == len(close_1h):
            rs_series = close_1h / np.where(btc_1h_close != 0, btc_1h_close, 1e-9)
            rs_strength = pct_change(rs_series, n=10)

        last_close = float(close_1h[-1])
        last_low = float(low_1h[-1])
        last_atr = float(atr_1h[-1])

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
        score += math.log1p(max(0.0, avg_vol_usdc) / 1e5)

        breakout_level = float(hh20[-1])
        last_high = float(high_1h[-1])
        confirm_price = last_high if RELAX_B_HIGH else last_close

        # Thresholds (relaxed mode widens)
        prox_min = PROX_ATR_MIN if not RELAXED_SIGNALS else min(-0.1, PROX_ATR_MIN)
        prox_max = PROX_ATR_MAX if not RELAXED_SIGNALS else max(1.0, PROX_ATR_MAX)
        vz_pre = VOL_Z_MIN_PRE if not RELAXED_SIGNALS else min(0.5, VOL_Z_MIN_PRE)
        vz_break = VOL_Z_MIN_BREAK if not RELAXED_SIGNALS else min(0.7, VOL_Z_MIN_BREAK)

        confirmed_breakout = (
            (trend_ok or (RELAXED_SIGNALS and lower_tf_ok))
            and confirm_price > (breakout_level + BREAK_BUFFER_ATR * last_atr)
            and vz >= vz_break
            and lower_tf_ok
        )
        pre_breakout = (
            (trend_ok or (RELAXED_SIGNALS and lower_tf_ok))
            and atr_rising
            and prox_min <= prox <= prox_max
            and vz >= vz_pre
        )

        # Levels
        entry = breakout_level + BREAK_BUFFER_ATR * last_atr
        base_stop = float(ll20[-1]) - 0.4 * last_atr  # a bit below LL20
        natural_risk = max(entry - base_stop, entry * 0.005)  # floor 0.5%
        stop = entry - natural_risk
        t1 = entry + 0.8 * last_atr
        t2 = entry + 1.5 * last_atr

        # ---------- SIZE FEASIBILITY (minQty / minNotional) ----------
        f = _SYMBOL_FILTERS.get(symbol) or {}
        min_qty = float(f.get("min_qty", 0.0))
        step_qty = float(f.get("step_qty", 0.0))
        min_notional = float(f.get("min_notional", MIN_NOTIONAL_FALLBACK))

        # advisory equity the scanner assumes
        equity = max(ANALYSES_EQUITY_USDC, 0.0)
        risk_dollars = equity * RISK_PER_TRADE
        rpu = max(entry - stop, entry * 0.002)  # risk per unit
        raw_qty = risk_dollars / rpu if rpu > 0 else 0.0
        q_qty = quantize_qty(raw_qty, step_qty if step_qty > 0 else 1e-8)

        too_small = (q_qty <= 0) or (q_qty < min_qty) or (entry * q_qty < max(MIN_NOTIONAL_FALLBACK, min_notional))

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
            "avg_vol": avg_vol_usdc,
            "advisory_qty": q_qty,
            "advisory_notional": entry * q_qty,
            "min_qty": min_qty,
            "min_notional": min_notional,
        }

        if too_small:
            out["signal"] = "N"
            out.setdefault("reasons", []).append("TooSmall(min)")
            return out

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
    with dest_latest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

# -------------------------- PIPELINE ----------------------------------

def compute_regime(btc_1h: Dict[str, np.ndarray], btc_4h: Dict[str, np.ndarray]) -> Tuple[bool, str]:
    try:
        ema200_4h = ema(btc_4h["close"], EMA_SLOW)
        ema20_4h = ema(btc_4h["close"], EMA_FAST)
        ema200_1h = ema(btc_1h["close"], EMA_SLOW)
        ema20_1h = ema(btc_1h["close"], EMA_FAST)
        if len(ema200_4h) > 4 and len(ema200_1h) > 4:
            slope4 = float(ema20_4h[-1] - ema20_4h[-4])
            slope1 = float(ema20_1h[-1] - ema20_1h[-4])
            if (float(btc_4h["close"][-1]) > float(ema200_4h[-1]) and
                float(btc_1h["close"][-1]) > float(ema200_1h[-1]) and
                slope4 > 0 and slope1 > 0):
                return True, "BTC uptrend (4h & 1h)"
            return False, "BTC not in uptrend"
        return False, "insufficient data"
    except Exception:
        return False, "regime calc error"

def run_pipeline(mode: str, ignore_regime: bool = False, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
    # load symbol filters once
    load_symbol_filters()

    symbols_meta = fetch_exchange_usdc_symbols()
    if not symbols_meta:
        return {
            "generated_at": human_time(),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "exchangeInfo unavailable"},
            "signals": {"type": "HOLD"},
            "orders": [], "candidates": [],
            "universe": {"scanned": 0, "eligible": 0, "skipped": {"no_data": [], "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode},
        }

    symbols = list(symbols_meta.keys())

    # Backtest path (daily end_time stepping)
    if start_date and end_date:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        current_dt = start_dt
        results = []
        while current_dt <= end_dt:
            results.append(process_day(symbols, mode, ignore_regime, int(current_dt.timestamp() * 1000)))
            current_dt += timedelta(days=1)
        stats = {
            "total_days": len(results),
            "total_breakouts": sum(len(p.get("orders", [])) for p in results),
            "avg_candidates": round(statistics.fmean(len(p.get("candidates", [])) for p in results), 3) if results else 0,
        }
        return {"backtest_results": results, "stats": stats}

    # Live path
    return process_day(symbols, mode, ignore_regime, None)

def process_day(symbols: List[str], mode: str, ignore_regime: bool, end_time: Optional[int]) -> dict:
    started = datetime.now(timezone.utc).astimezone(TZ) if not end_time else datetime.fromtimestamp(end_time / 1000, timezone.utc).astimezone(TZ)

    # BTC baseline
    btc_raw_1h = fetch_klines("BTCUSDC", INTERVALS["1h"][0], INTERVALS["1h"][1], end_time)
    btc_raw_4h = fetch_klines("BTCUSDC", INTERVALS["4h"][0], INTERVALS["4h"][1], end_time)
    if not btc_raw_1h or not btc_raw_4h:
        return {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "BTC data unavailable"},
            "signals": {"type": "HOLD"},
            "orders": [], "candidates": [],
            "universe": {"scanned": 0, "eligible": len(symbols), "skipped": {"no_data": [], "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode},
        }
    btc_1h = parse_klines(btc_raw_1h); btc_4h = parse_klines(btc_raw_4h)
    regime_ok, regime_reason = compute_regime(btc_1h, btc_4h)
    if ignore_regime:
        regime_ok, regime_reason = True, "Regime ignored by flag"
    btc_close_1h = btc_1h["close"]

    # Thin for light modes
    all_symbols = symbols[:]
    if mode == "light-fast":
        random.shuffle(all_symbols); all_symbols = all_symbols[: len(all_symbols)//2]
    elif mode == "light-hourly":
        random.shuffle(all_symbols); all_symbols = all_symbols[: len(all_symbols)//3]

    # Fetch all TFs in parallel per TF
    reqs: Dict[str, Dict[str, Optional[Dict[str, np.ndarray]]]] = {}
    for tf, (interval, limit) in INTERVALS.items():
        raw_all = fetch_all_klines(all_symbols, interval, limit, end_time)
        reqs[tf] = {sym: (parse_klines(raw_all[sym]) if raw_all[sym] else None) for sym in all_symbols}

    candidates: List[dict] = []
    confirmed: List[dict] = []
    no_data_syms: List[str] = []
    all_infos: List[dict] = []
    scanned = 0

    for sym in all_symbols:
        scanned += 1
        tf_pack = {tf: reqs[tf][sym] for tf in INTERVALS}
        if any(d is None for d in tf_pack.values()):
            no_data_syms.append(sym.split("USDC")[0])
            continue
        info = analyze_symbol(sym, btc_close_1h, tf_pack)
        if not info:
            no_data_syms.append(sym.split("USDC")[0])
            continue
        info["ticker"] = sym.split("USDC")[0]
        all_infos.append(info)
        if info["signal"] == "B":
            confirmed.append(info)
        elif info["signal"] == "C":
            candidates.append(info)

    # Near-miss logging
    non_signals = sorted([i for i in all_infos if i.get("signal") == "N"],
                         key=lambda x: x.get("score", 0.0), reverse=True)
    for i in range(min(NEAR_MISS_LOG_COUNT, len(non_signals))):
        ns = non_signals[i]
        logging.info(f"Near-miss {ns['symbol']}: score={ns.get('score',0):.2f}, "
                     f"vz={ns.get('vol_z',0):.2f}, prox={ns.get('prox_atr',0):.2f}, "
                     f"reasons={ns.get('reasons',[])}")

    if not regime_ok:
        return {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": regime_ok, "reason": regime_reason},
            "signals": {"type": "HOLD"},
            "orders": [], "candidates": [],
            "universe": {"scanned": scanned, "eligible": len(all_symbols),
                         "skipped": {"no_data": sorted(no_data_syms), "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode},
        }

    # Rank & payloads
    candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    top_candidates = candidates[:MAX_CANDIDATES]

    signal_type = "HOLD"
    orders: List[dict] = []
    if confirmed:
        signal_type = "B"
        for o in confirmed:
            orders.append({
                "ticker": o["ticker"], "symbol": o["symbol"],
                "entry": round(float(o["entry"]), 8),
                "stop": round(float(o["stop"]), 8),
                "t1": round(float(o["t1"]), 8),
                "t2": round(float(o["t2"]), 8),
                "atr": round(float(o["atr"]), 8),
                "tf": "1h",
                "notes": o.get("reasons", []),
                "rotation_exempt": False,
                "advisory_qty": round(float(o.get("advisory_qty", 0.0)), 8),
                "advisory_notional": round(float(o.get("advisory_notional", 0.0)), 8),
            })
    elif top_candidates:
        signal_type = "C"

    cand_payload = [{
        "ticker": c["ticker"], "symbol": c["symbol"],
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
        "advisory_qty": round(float(c.get("advisory_qty", 0.0)), 8),
        "advisory_notional": round(float(c.get("advisory_notional", 0.0)), 8),
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
            "eligible": len(all_symbols),
            "skipped": {"no_data": sorted(no_data_syms), "missing_binance": []},
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
                "CAPITAL": CAPITAL, "RISK_PER_TRADE": RISK_PER_TRADE,
                "ANALYSES_EQUITY_USDC": ANALYSES_EQUITY_USDC,
                "MIN_NOTIONAL_FALLBACK": MIN_NOTIONAL_FALLBACK,
                "RELAXED_SIGNALS": RELAXED_SIGNALS,
            },
            "lower_tf_confirm": True,
            "rs_reference": "BTCUSDC",
            "binance_endpoint": BASE_URL,
            "mode": mode,
        }
    }
    return payload

# ------------------------------ CLI -----------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Analyses pipeline (USDC)")
    p.add_argument("--mode", default="deep", choices=["deep", "light-fast", "light-hourly"])
    p.add_argument("--ignore_regime", action="store_true")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--start_date", type=str)
    p.add_argument("--end_date", type=str)
    args = p.parse_args()

    if args.backtest and (not args.start_date or not args.end_date):
        logging.error("Backtest requires --start_date and --end_date")
        return

    try:
        payload = (run_pipeline(args.mode, args.ignore_regime, args.start_date, args.end_date)
                   if args.backtest else
                   run_pipeline(args.mode, args.ignore_regime))
    except Exception:
        logging.error("UNCAUGHT ERROR:\n" + traceback.format_exc())
        payload = {
            "generated_at": human_time(),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "uncaught error"},
            "signals": {"type": "HOLD"},
            "orders": [], "candidates": [],
            "universe": {"scanned": 0, "eligible": 0, "skipped": {"no_data": [], "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": args.mode},
        }

    latest_path = Path("public_runs/latest/summary.json")
    write_summary(payload, latest_path)
    stamp = datetime.now(timezone.utc).astimezone(TZ).strftime("%Y%m%d_%H%M%S")
    snap = Path("public_runs") / stamp
    ensure_dirs(snap / "summary.json")
    with (snap / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

    logging.info(f"Summary written to {latest_path} (signal={payload.get('signals',{}).get('type')}, "
                 f"regime_ok={payload.get('regime',{}).get('ok')})")

if __name__ == "__main__":
    main()