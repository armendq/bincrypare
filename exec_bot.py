#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Execution bot for Binance Spot (USDC quote)
- Reads summary.json (local file or URL)
- Places stop-limit BUYs for C candidates (optional)
- Places market BUYs for B breakouts
- Places two TP LIMITs after entry
- Tracks open state in state/open_positions.json
- Promotes pending -> live when exchange order fills
- Skips any order that breaches minNotional before submitting
"""

import os, json, time, math, requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, Optional

from binance.client import Client
from binance.enums import (
    SIDE_BUY, SIDE_SELL, TIME_IN_FORCE_GTC,
    ORDER_TYPE_MARKET, ORDER_TYPE_STOP_LOSS_LIMIT, ORDER_TYPE_LIMIT
)

# -------------------- ENV / CONFIG --------------------

SUMMARY_URL        = os.getenv("SUMMARY_URL", "https://raw.githubusercontent.com/armendq/bincrypare/main/public_runs/latest/summary.json")
QUOTE_ASSET        = os.getenv("QUOTE_ASSET", "USDC")
DRY_RUN            = os.getenv("DRY_RUN", "1") == "1"

# sizing
FORCE_EQUITY_USD   = float(os.getenv("FORCE_EQUITY_USD", "0"))  # if >0, overrides live free balance
RISK_PCT           = float(os.getenv("RISK_PCT", "0.012"))       # 1.2% risk/trade baseline

# candidates trading
TRADE_CANDIDATES   = os.getenv("TRADE_CANDIDATES", "1") == "1"
C_RISK_MULT        = float(os.getenv("C_RISK_MULT", "0.5"))      # risk fraction for C
STOP_LIMIT_OFFSET  = float(os.getenv("STOP_LIMIT_OFFSET", "0.001"))  # +0.1% above stop-price

# min notional handling
MIN_NOTIONAL_FALLBACK = float(os.getenv("MIN_NOTIONAL", "5.0"))
ALLOW_MIN_ORDER    = os.getenv("ALLOW_MIN_ORDER", "0") == "1"    # allow micro order fallback
MIN_ORDER_USD      = float(os.getenv("MIN_ORDER_USD", "6"))      # at least this much if fallback used

# lifecycle rules (timeouts)
PENDING_MAX_MINUTES = int(os.getenv("PENDING_MAX_MINUTES", "180"))  # cancel pendings older than this
LIVE_MAX_HOURS      = int(os.getenv("LIVE_MAX_HOURS", "12"))        # optional time-stop for live
ENABLE_TIME_STOP    = os.getenv("ENABLE_TIME_STOP", "1") == "1"     # enable the live position time-stop rule

API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

STATE_DIR  = Path("state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "open_positions.json"

client = Client(API_KEY, API_SECRET)

# -------------------- UTIL --------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def now_str() -> str:
    return now_utc().astimezone().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str) -> None:
    print(f"[exec {now_str()}] {msg}", flush=True)

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text("utf-8"))
        except Exception:
            return {}
    return {}

def save_state(state: dict) -> None:
    tmp = STATE_DIR / "open_positions.tmp"
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), "utf-8")
    tmp.replace(STATE_FILE)

def fetch_summary() -> Optional[dict]:
    # local file path support
    if SUMMARY_URL.startswith("file://"):
        try:
            p = SUMMARY_URL.replace("file://", "", 1)
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log(f"[LOCAL READ ERROR] {e}")
            return None
    # http(s)
    for attempt in (1, 2):
        try:
            r = requests.get(SUMMARY_URL, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(3)
    log("Hold and wait. Fetch failed twice.")
    return None

# -------------- BINANCE FILTERS & ROUNDING --------------

def get_symbol_filters(symbol: str) -> Tuple[float, float, float, float]:
    """
    Returns (minQty, stepSize, tickSize, minNotional)
    """
    info = client.get_symbol_info(symbol)
    lot = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
    pricef = next(f for f in info["filters"] if f["filterType"] == "PRICE_FILTER")
    notional = next((f for f in info["filters"] if f["filterType"] in ("NOTIONAL", "MIN_NOTIONAL")), {})
    min_notional = float(notional.get("minNotional", MIN_NOTIONAL_FALLBACK)) if notional else MIN_NOTIONAL_FALLBACK
    return float(lot["minQty"]), float(lot["stepSize"]), float(pricef["tickSize"]), max(MIN_NOTIONAL_FALLBACK, min_notional)

def floor_to_step(v: float, step: float) -> float:
    if step <= 0:
        return v
    # use decimal-like floor by integer division
    return math.floor(v / step) * step

def round_to_tick(p: float, tick: float) -> float:
    if tick <= 0:
        return p
    return floor_to_step(p, tick)

# -------------------- SIZING --------------------

def compute_qty(entry: float, stop: float, equity: float,
                min_qty: float, step_qty: float, min_notional: float,
                price_for_notional: float,
                allow_min_order: bool) -> Tuple[float, str]:
    """
    Returns (qty, reason)
      - qty == 0 means 'do not trade' with reason
      - qty > 0 is rounded to step and passes notional/lot minimums
    """
    # risk dollars
    risk_dollars = max(equity * RISK_PCT, 0.0)

    # risk per unit (distance from entry to stop, but not less than 0.2%)
    rpu = max(entry - stop, entry * 0.002)
    if rpu <= 0:
        return 0.0, "bad_risk_params"

    raw_qty = risk_dollars / rpu

    # ensure meets minNotional
    min_qty_by_notional = min_notional / max(price_for_notional, 1e-12)
    raw_qty = max(raw_qty, min_qty_by_notional)

    qty = floor_to_step(raw_qty, step_qty)

    # If still below minQty, optionally try a micro fallback based on MIN_ORDER_USD
    if qty < min_qty:
        if allow_min_order:
            micro = MIN_ORDER_USD / max(price_for_notional, 1e-12)
            qty2 = floor_to_step(max(micro, min_qty), step_qty)
            notional2 = qty2 * price_for_notional
            if qty2 >= min_qty and notional2 >= min_notional:
                return qty2, "fallback_min_order"
        return 0.0, "below_lot_size"

    # final notional guard
    if qty * price_for_notional < min_notional:
        if allow_min_order:
            micro = MIN_ORDER_USD / max(price_for_notional, 1e-12)
            qty2 = floor_to_step(max(qty, micro), step_qty)
            if qty2 * price_for_notional >= min_notional and qty2 >= min_qty:
                return qty2, "fallback_notional"
        return 0.0, "below_notional"

    return qty, "ok"

# -------------------- ORDER WRAPPERS --------------------

def place_market_buy(symbol: str, qty: float) -> Optional[dict]:
    if DRY_RUN:
        log(f"[MARKET BUY dry] {symbol} qty={qty:.8f}")
        return {"orderId": "dry"}
    try:
        o = client.create_order(symbol=symbol, side=SIDE_BUY,
                                type=ORDER_TYPE_MARKET, quantity=qty)
        log(f"[MARKET BUY] {symbol} qty={qty:.8f} id={o.get('orderId')}")
        return o
    except Exception as e:
        log(f"[ENTRY ERROR] {symbol}: {e}")
        return None

def place_stop_limit_buy(symbol: str, qty: float, entry: float, tick: float) -> Optional[dict]:
    """
    Use stopPrice = rounded entry, limitPrice slightly above (by STOP_LIMIT_OFFSET)
    """
    stop_px  = round_to_tick(entry, tick)
    limit_px = round_to_tick(entry * (1 + STOP_LIMIT_OFFSET), tick)
    if DRY_RUN:
        log(f"[STOP-LIMIT BUY dry] {symbol} qty={qty:.8f} stop={stop_px} limit={limit_px}")
        return {"orderId": "dry"}
    try:
        o = client.create_order(
            symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_STOP_LOSS_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC, quantity=qty,
            price=f"{limit_px:.08f}", stopPrice=f"{stop_px:.08f}"
        )
        log(f"[STOP-LIMIT BUY] {symbol} qty={qty:.8f} stop={stop_px} limit={limit_px} id={o.get('orderId')}")
        return o
    except Exception as e:
        log(f"[STOP-LIMIT ERROR] {symbol}: {e}")
        return None

def place_tp_limits(symbol: str, qty: float, t1: float, t2: float, tick: float) -> Tuple[Optional[dict], Optional[dict]]:
    q1 = floor_to_step(qty * 0.5, 1e-8)
    q2 = max(qty - q1, 0.0)
    p1 = round_to_tick(t1, tick)
    p2 = round_to_tick(t2, tick)
    if DRY_RUN:
        log(f"[TP dry] {symbol} sell {q1:.8f}@{p1} and {q2:.8f}@{p2}")
        return {"orderId": "dry"}, {"orderId": "dry"}
    try:
        o1 = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT,
                                 timeInForce=TIME_IN_FORCE_GTC, quantity=q1, price=f"{p1:.08f}")
        o2 = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT,
                                 timeInForce=TIME_IN_FORCE_GTC, quantity=q2, price=f"{p2:.08f}")
        log(f"[TP] {symbol} t1 {q1:.8f}@{p1} id={o1.get('orderId')}; t2 {q2:.8f}@{p2} id={o2.get('orderId')}")
        return o1, o2
    except Exception as e:
        log(f"[TP ERROR] {symbol}: {e}")
        return None, None

# -------------------- ACCOUNT & EQUITY --------------------

def get_spot_balance(asset: str) -> Tuple[float, float, float]:
    try:
        acct = client.get_account()
    except Exception as e:
        log(f"[ACCOUNT ERROR] {e}")
        return 0.0, 0.0, 0.0
    free = locked = 0.0
    for b in acct.get("balances", []):
        if b.get("asset") == asset:
            free  = float(b.get("free", "0"))
            locked= float(b.get("locked", "0"))
            break
    return free, locked, free + locked

def equity_for_sizing() -> float:
    if FORCE_EQUITY_USD > 0:
        return FORCE_EQUITY_USD
    free, _, _ = get_spot_balance(QUOTE_ASSET)
    return free

# -------------------- ORDER MANAGEMENT --------------------

def fetch_avg_price(symbol: str) -> Optional[float]:
    """Lightweight price for notional checks if needed."""
    try:
        d = client.get_symbol_ticker(symbol=symbol)
        return float(d["price"])
    except Exception:
        return None

def poll_order(symbol: str, order_id: Any) -> Optional[dict]:
    try:
        return client.get_order(symbol=symbol, orderId=order_id)
    except Exception:
        return None

def cancel_order(symbol: str, order_id: Any) -> bool:
    if DRY_RUN:
        log(f"[CANCEL dry] {symbol} id={order_id}")
        return True
    try:
        client.cancel_order(symbol=symbol, orderId=order_id)
        log(f"[CANCEL] {symbol} id={order_id}")
        return True
    except Exception as e:
        log(f"[CANCEL ERROR] {symbol}: {e}")
        return False

# -------------------- MAIN --------------------

def main():
    free, locked, total = get_spot_balance(QUOTE_ASSET)
    if FORCE_EQUITY_USD > 0:
        log(f"Balance: OVERRIDE {FORCE_EQUITY_USD:.2f} {QUOTE_ASSET} | live free={free:.2f}, locked={locked:.2f}, total={total:.2f}")
    else:
        log(f"Balance: live free={free:.2f}, locked={locked:.2f}, total={total:.2f} {QUOTE_ASSET}")

    summary = fetch_summary()
    if not summary:
        return

    # ---- state load/update header
    state = load_state()
    state["_last_run"] = now_str()
    state["_last_balance"] = {"asset": QUOTE_ASSET, "free": round(free, 8), "locked": round(locked, 8), "total": round(total, 8)}

    # ---- 1) Promote or prune existing state first
    # Promote filled pendings -> live and place TPs
    to_delete = []
    for sym, pos in list(state.items()):
        if sym.startswith("_"):
            continue
        typ   = pos.get("type")
        stat  = pos.get("status")
        oid   = pos.get("entry_order_id")
        ctime = pos.get("created_at")

        # Expire very old pendings
        if stat == "pending" and PENDING_MAX_MINUTES > 0 and ctime:
            try:
                age_min = (now_utc().timestamp() - float(ctime)) / 60.0
                if age_min > PENDING_MAX_MINUTES and oid:
                    cancel_order(sym, oid)
                    log(f"[PENDING TIMEOUT] {sym} > {PENDING_MAX_MINUTES}m -> cancelled & removed")
                    to_delete.append(sym)
                    continue
            except Exception:
                pass

        # Check order status for pending
        if stat == "pending" and oid:
            od = poll_order(sym, oid)
            if od and od.get("status") in ("FILLED", "PARTIALLY_FILLED"):
                filled_qty = float(od.get("executedQty", "0"))
                if filled_qty > 0:
                    # place TPs now
                    tick = float(pos.get("tick", "0"))
                    if tick <= 0:
                        # safety: fetch from filters if missing
                        try:
                            _, _, tick, _ = get_symbol_filters(sym)
                        except Exception:
                            tick = 0.0
                    place_tp_limits(sym, filled_qty, float(pos["t1"]), float(pos["t2"]), tick if tick > 0 else 1e-8)
                    pos["filled_qty"] = filled_qty
                    pos["status"] = "live"
                    pos["activated_at"] = now_utc().timestamp()
                    log(f"[PROMOTE] {sym} pending -> live (qty={filled_qty})")

        # Optional time-stop on live positions (close after LIVE_MAX_HOURS)
        if ENABLE_TIME_STOP and stat == "live" and LIVE_MAX_HOURS > 0:
            try:
                act_ts = float(pos.get("activated_at", 0))
                if act_ts > 0 and (now_utc().timestamp() - act_ts) > LIVE_MAX_HOURS * 3600:
                    # place a market sell to free funds (best effort)
                    qty = float(pos.get("filled_qty", 0.0))
                    if qty > 0 and not DRY_RUN:
                        try:
                            client.create_order(symbol=sym, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=qty)
                            log(f"[TIME-STOP] {sym} > {LIVE_MAX_HOURS}h -> market sell {qty}")
                        except Exception as e:
                            log(f"[TIME-STOP SELL ERROR] {sym}: {e}")
                    to_delete.append(sym)
            except Exception:
                pass

    for k in to_delete:
        state.pop(k, None)

    # ---- 2) New orders from summary
    # Candidates -> stop-limit
    if TRADE_CANDIDATES:
        eq_c = equity_for_sizing() * max(0.0, C_RISK_MULT)
        for c in summary.get("candidates", []):
            sym = c["symbol"]
            if not sym.endswith(QUOTE_ASSET):
                continue
            entry = float(c["entry"]); stop = float(c["stop"]); t1=float(c["t1"]); t2=float(c["t2"])

            # filters
            try:
                min_qty, step_qty, tick, min_notional = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}")
                continue

            # pre-validate notional with LIMIT price (Binance evaluates notional at order price)
            limit_price = round_to_tick(entry * (1 + STOP_LIMIT_OFFSET), tick)
            qty, why = compute_qty(entry, stop, eq_c, min_qty, step_qty, min_notional,
                                   price_for_notional=limit_price,
                                   allow_min_order=ALLOW_MIN_ORDER)
            if qty <= 0:
                log(f"[CAND SKIP] {sym} qty={qty} reason={why}")
                continue

            # final safety: notional check
            if qty * limit_price < min_notional:
                log(f"[SKIP NOTIONAL] {sym} notional={qty*limit_price:.4f} < minNotional={min_notional}")
                continue

            # send order
            o = place_stop_limit_buy(sym, qty, entry, tick)
            if o:
                state[sym] = {
                    "entry": entry, "stop": stop, "t1": t1, "t2": t2,
                    "filled_qty": 0.0, "t1_filled_qty": 0.0,
                    "status": "pending", "entry_order_id": o.get("orderId"),
                    "type": "C",
                    "tick": tick,
                    "min_qty": min_qty,
                    "step_qty": step_qty,
                    "min_notional": min_notional,
                    "created_at": now_utc().timestamp()
                }

    # Breakouts -> market buy + TPs
    if summary.get("signals", {}).get("type") == "B":
        eq_b = equity_for_sizing()
        for o in summary.get("orders", []) or []:
            sym = o["symbol"]
            if not sym.endswith(QUOTE_ASSET):
                continue
            entry = float(o["entry"]); stop = float(o["stop"]); t1=float(o["t1"]); t2=float(o["t2"])

            # filters
            try:
                min_qty, step_qty, tick, min_notional = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}")
                continue

            # For market orders, estimate notional with latest price (or entry)
            mprice = fetch_avg_price(sym) or entry
            qty, why = compute_qty(entry, stop, eq_b, min_qty, step_qty, min_notional,
                                   price_for_notional=mprice,
                                   allow_min_order=ALLOW_MIN_ORDER)
            if qty <= 0:
                log(f"[BREAKOUT SKIP] {sym} qty={qty} reason={why}")
                continue

            if qty * mprice < min_notional:
                log(f"[SKIP NOTIONAL] {sym} notional={qty*mprice:.4f} < minNotional={min_notional}")
                continue

            m = place_market_buy(sym, qty)
            if m:
                place_tp_limits(sym, qty, t1, t2, tick)
                state[sym] = {
                    "entry": entry, "stop": stop, "t1": t1, "t2": t2,
                    "filled_qty": qty, "t1_filled_qty": 0.0,
                    "status": "live", "entry_order_id": m.get("orderId"),
                    "type": "B",
                    "tick": tick,
                    "min_qty": min_qty,
                    "step_qty": step_qty,
                    "min_notional": min_notional,
                    "created_at": now_utc().timestamp(),
                    "activated_at": now_utc().timestamp()
                }

    save_state(state)

if __name__ == "__main__":
    main()