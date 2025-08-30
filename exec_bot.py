#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Executor bot (USDC spot)

- Reads summary.json (local file URL by default)
- Places STOP-LOSS-LIMIT buys for "C" candidates (reduced risk slice)
- Places MARKET buys + TP LIMITs for "B" breakouts
- Uses Decimal for exchange precision and rounds down to lot/price step
- Optional fallback: if risk-based qty is below Binance minimums, place the
  smallest compliant order (controlled by ALLOW_MIN_ORDER/MIN_ORDER_USD)
"""

import os, json, time, math, requests
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, getcontext, ROUND_DOWN

from binance.client import Client
from binance.enums import (
    SIDE_BUY, SIDE_SELL, TIME_IN_FORCE_GTC,
    ORDER_TYPE_MARKET, ORDER_TYPE_STOP_LOSS_LIMIT, ORDER_TYPE_LIMIT
)

# ---------- Decimal helpers ----------
getcontext().prec = 28
def d(x) -> Decimal:
    if isinstance(x, Decimal): return x
    return Decimal(str(x))

def fmt_dec(x: Decimal) -> str:
    """Trim trailing zeros while keeping decimal form Binance accepts."""
    s = f"{x:f}"
    # Binance is happy with e.g. "1", but we'll ensure at least plain decimal
    return s

def quantize_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Floor value to the nearest step (LOT_SIZE or PRICE tick)."""
    if step <= 0: 
        return value
    # (value // step) * step floors to step
    return (value // step) * step

# ---------- ENV ----------
SUMMARY_URL        = os.getenv("SUMMARY_URL", "https://raw.githubusercontent.com/armendq/bincrypare/main/public_runs/latest/summary.json")
QUOTE_ASSET        = os.getenv("QUOTE_ASSET", "USDC")
DRY_RUN            = os.getenv("DRY_RUN", "1") == "1"
FORCE_EQUITY       = d(os.getenv("FORCE_EQUITY_USD", "0"))
TRADE_CANDS        = os.getenv("TRADE_CANDIDATES", "1") == "1"
C_RISK_MULT        = d(os.getenv("C_RISK_MULT", "0.5"))
RISK_PCT           = d(os.getenv("RISK_PCT", "0.012"))
STOP_LIMIT_OFFSET  = d(os.getenv("STOP_LIMIT_OFFSET", "0.001"))
MIN_NOTIONAL_FALLBACK = d(os.getenv("MIN_NOTIONAL", "5"))

# New: fallback control
ALLOW_MIN_ORDER    = os.getenv("ALLOW_MIN_ORDER", "1") == "1"
MIN_ORDER_USD      = d(os.getenv("MIN_ORDER_USD", "6"))

API_KEY            = os.getenv("BINANCE_API_KEY", "")
API_SECRET         = os.getenv("BINANCE_API_SECRET", "")

STATE_DIR  = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "open_positions.json"

client = Client(API_KEY, API_SECRET)

# ---------- Logging ----------
def now_str() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str) -> None:
    print(f"[exec {now_str()}] {msg}", flush=True)

# ---------- State ----------
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

# ---------- IO ----------
def fetch_summary() -> dict | None:
    if SUMMARY_URL.startswith("file://"):
        try:
            p = SUMMARY_URL.replace("file://", "", 1)
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log(f"[LOCAL READ ERROR] {e}")
            return None
    for attempt in (1, 2):
        try:
            r = requests.get(SUMMARY_URL, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(15)
    log("Hold and wait. Fetch failed twice.")
    return None

# ---------- Exchange helpers ----------
def get_spot_balance(asset: str) -> tuple[Decimal, Decimal, Decimal]:
    try:
        acct = client.get_account()
    except Exception as e:
        log(f"[ACCOUNT ERROR] {e}")
        return d(0), d(0), d(0)
    free = locked = d(0)
    for b in acct.get("balances", []):
        if b.get("asset") == asset:
            free  = d(b.get("free", "0"))
            locked= d(b.get("locked", "0"))
            break
    return free, locked, free + locked

def equity_for_sizing() -> Decimal:
    if FORCE_EQUITY > 0:
        return FORCE_EQUITY
    free, _, _ = get_spot_balance(QUOTE_ASSET)
    return free

def get_symbol_filters(symbol: str) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """
    Returns (minQty, stepSize, tickSize, minNotional) as Decimals.
    """
    info = client.get_symbol_info(symbol)
    lot = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
    pricef = next(f for f in info["filters"] if f["filterType"] == "PRICE_FILTER")
    notional = next((f for f in info["filters"] if f["filterType"] in ("NOTIONAL","MIN_NOTIONAL")), {})
    min_notional = d(notional.get("minNotional", str(MIN_NOTIONAL_FALLBACK))) if notional else MIN_NOTIONAL_FALLBACK
    return d(lot["minQty"]), d(lot["stepSize"]), d(pricef["tickSize"]), max(MIN_NOTIONAL_FALLBACK, min_notional)

# ---------- Sizing ----------
def compute_qty(entry: Decimal, stop: Decimal, equity: Decimal,
                min_qty: Decimal, step_qty: Decimal, min_notional: Decimal) -> Decimal:
    """
    Primary: risk-based sizing (risk = equity * RISK_PCT).
    Fallback (if too small): place the smallest order that satisfies BOTH minQty and minNotional,
    capped by available equity. Controlled by ALLOW_MIN_ORDER/MIN_ORDER_USD.
    """
    # ---- risk-based size ----
    risk_dollars = equity * RISK_PCT
    rpu = max(entry - stop, entry * d("0.002"))   # guard if stop is very tight
    if rpu > 0:
        # Make sure we also respect minNotional here
        raw_qty = max(risk_dollars / rpu, min_notional / max(entry, d("1e-12")))
        qty = quantize_to_step(raw_qty, step_qty)
        if qty >= min_qty and qty * entry >= min_notional and qty * entry <= equity:
            return qty

    # ---- fallback: smallest compliant order ----
    if not ALLOW_MIN_ORDER:
        return d(0)

    target_notional = max(min_notional, MIN_ORDER_USD)
    q_min_by_notional = quantize_to_step((target_notional / entry), step_qty)
    q_fallback = max(min_qty, q_min_by_notional)

    # ensure we can actually afford it
    if q_fallback * entry > equity:
        return d(0)

    return q_fallback

# ---------- Orders ----------
def place_market_buy(symbol: str, qty: Decimal) -> dict | None:
    qty_str = fmt_dec(qty)
    if DRY_RUN:
        log(f"[MARKET BUY dry] {symbol} qty={qty_str}")
        return {"orderId": "dry"}
    try:
        o = client.create_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=qty_str)
        log(f"[MARKET BUY] {symbol} qty={qty_str} id={o.get('orderId')}")
        return o
    except Exception as e:
        log(f"[ENTRY ERROR] {symbol}: {e}")
        return None

def place_stop_limit_buy(symbol: str, qty: Decimal, entry: Decimal, tick: Decimal) -> dict | None:
    stop_px  = quantize_to_step(entry, tick)
    limit_px = quantize_to_step(entry * (d("1") + STOP_LIMIT_OFFSET), tick)
    qty_str  = fmt_dec(qty)
    stop_str = fmt_dec(stop_px)
    limit_str= fmt_dec(limit_px)

    if DRY_RUN:
        log(f"[STOP-LIMIT BUY dry] {symbol} qty={qty_str} stop={stop_str} limit={limit_str}")
        return {"orderId": "dry"}
    try:
        o = client.create_order(
            symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_STOP_LOSS_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC, quantity=qty_str,
            price=limit_str, stopPrice=stop_str
        )
        log(f"[STOP-LIMIT BUY] {symbol} qty={qty_str} stop={stop_str} limit={limit_str} id={o.get('orderId')}")
        return o
    except Exception as e:
        log(f"[STOP-LIMIT ERROR] {symbol}: {e}")
        return None

def place_tp_limits(symbol: str, qty: Decimal, t1: Decimal, t2: Decimal, tick: Decimal) -> None:
    q1 = quantize_to_step(qty * d("0.5"), d("0.00000001"))
    q2 = max(qty - q1, d("0"))
    p1 = quantize_to_step(t1, tick)
    p2 = quantize_to_step(t2, tick)
    if DRY_RUN:
        log(f"[TP dry] {symbol} sell {fmt_dec(q1)}@{fmt_dec(p1)} and {fmt_dec(q2)}@{fmt_dec(p2)}")
        return
    try:
        o1 = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT,
                                 timeInForce=TIME_IN_FORCE_GTC, quantity=fmt_dec(q1), price=fmt_dec(p1))
        o2 = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT,
                                 timeInForce=TIME_IN_FORCE_GTC, quantity=fmt_dec(q2), price=fmt_dec(p2))
        log(f"[TP] {symbol} t1 {fmt_dec(q1)}@{fmt_dec(p1)} id={o1.get('orderId')}; "
            f"t2 {fmt_dec(q2)}@{fmt_dec(p2)} id={o2.get('orderId')}")
    except Exception as e:
        log(f"[TP ERROR] {symbol}: {e}")

# ---------- Main ----------
def main():
    free, locked, total = get_spot_balance(QUOTE_ASSET)
    if FORCE_EQUITY > 0:
        log(f"Balance: OVERRIDE {fmt_dec(FORCE_EQUITY)} {QUOTE_ASSET} | live free={fmt_dec(free)}, "
            f"locked={fmt_dec(locked)}, total={fmt_dec(total)}")
    else:
        log(f"Balance: live free={fmt_dec(free)}, locked={fmt_dec(locked)}, total={fmt_dec(total)} {QUOTE_ASSET}")

    S = fetch_summary()
    if not S:
        return

    state = load_state()
    state["_last_run"] = now_str()
    state["_last_balance"] = {
        "asset": QUOTE_ASSET,
        "free": float(free), "locked": float(locked), "total": float(total)
    }

    # Candidates -> stop-limit buy at entry (reduced risk)
    if TRADE_CANDS:
        eq_c = equity_for_sizing() * max(d("0"), C_RISK_MULT)
        for c in S.get("candidates", []):
            sym = c.get("symbol")
            if not sym or not sym.endswith(QUOTE_ASSET): 
                continue
            entry = d(c["entry"]); stop = d(c["stop"]); t1 = d(c["t1"]); t2 = d(c["t2"])

            try:
                min_qty, step_qty, tick, min_notional = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}")
                continue

            qty = compute_qty(entry, stop, eq_c, min_qty, step_qty, min_notional)
            if qty <= 0:
                log(f"[CAND SKIP] {sym} qty too small after sizing; "
                    f"minQty={fmt_dec(min_qty)}, minNotional={fmt_dec(min_notional)}, "
                    f"entry≈{fmt_dec(entry)}, equity={fmt_dec(eq_c)}")
                continue

            o = place_stop_limit_buy(sym, qty, entry, tick)
            if o:
                state[sym] = {
                    "entry": float(entry), "stop": float(stop), "t1": float(t1), "t2": float(t2),
                    "filled_qty": 0.0, "t1_filled_qty": 0.0,
                    "status": "pending", "entry_order_id": o.get("orderId"),
                    "type": "C"
                }

    # Breakouts -> market buy now + place TPs
    if S.get("signals", {}).get("type") == "B":
        eq_b = equity_for_sizing()
        for o in S.get("orders", []) or []:
            sym = o.get("symbol")
            if not sym or not sym.endswith(QUOTE_ASSET):
                continue
            entry = d(o["entry"]); stop = d(o["stop"]); t1 = d(o["t1"]); t2 = d(o["t2"])

            try:
                min_qty, step_qty, tick, min_notional = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}")
                continue

            qty = compute_qty(entry, stop, eq_b, min_qty, step_qty, min_notional)
            if qty <= 0:
                log(f"[BREAKOUT SKIP] {sym} qty too small after sizing; "
                    f"minQty={fmt_dec(min_qty)}, minNotional={fmt_dec(min_notional)}, "
                    f"entry≈{fmt_dec(entry)}, equity={fmt_dec(eq_b)}")
                continue

            m = place_market_buy(sym, qty)
            if m:
                place_tp_limits(sym, qty, t1, t2, tick)
                state[sym] = {
                    "entry": float(entry), "stop": float(stop), "t1": float(t1), "t2": float(t2),
                    "filled_qty": float(qty), "t1_filled_qty": 0.0,
                    "status": "live", "entry_order_id": m.get("orderId"),
                    "type": "B"
                }

    save_state(state)

if __name__ == "__main__":
    main()