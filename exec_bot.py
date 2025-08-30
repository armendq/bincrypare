#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, math, requests
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, getcontext

from binance.client import Client
from binance.enums import (
    SIDE_BUY, SIDE_SELL, TIME_IN_FORCE_GTC,
    ORDER_TYPE_MARKET, ORDER_TYPE_STOP_LOSS_LIMIT, ORDER_TYPE_LIMIT
)

# ---- Decimal context (plenty of precision, we always ROUND_DOWN to meet filters)
getcontext().prec = 28

# -------- env
SUMMARY_URL       = os.getenv("SUMMARY_URL", "https://raw.githubusercontent.com/armendq/bincrypare/main/public_runs/latest/summary.json")
QUOTE_ASSET       = os.getenv("QUOTE_ASSET", "USDC")
DRY_RUN           = os.getenv("DRY_RUN", "1") == "1"
FORCE_EQUITY      = Decimal(os.getenv("FORCE_EQUITY_USD", "0"))
TRADE_CANDS       = os.getenv("TRADE_CANDIDATES", "1") == "1"
C_RISK_MULT       = Decimal(os.getenv("C_RISK_MULT", "0.5"))
RISK_PCT          = Decimal(os.getenv("RISK_PCT", "0.012"))
STOP_LIMIT_OFFSET = Decimal(os.getenv("STOP_LIMIT_OFFSET", "0.001"))
MIN_NOTIONAL_FALLBACK = Decimal(os.getenv("MIN_NOTIONAL", "5.0"))

API_KEY   = os.getenv("BINANCE_API_KEY", "")
API_SECRET= os.getenv("BINANCE_API_SECRET", "")

STATE_DIR  = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "open_positions.json"

client = Client(API_KEY, API_SECRET)

# -------- utils
def now_str() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str) -> None:
    print(f"[exec {now_str()}] {msg}", flush=True)

def d(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))

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

# ----- Binance filters & quantization helpers
def get_symbol_filters(symbol: str) -> tuple[Decimal, Decimal, Decimal, Decimal, int, int]:
    info = client.get_symbol_info(symbol)
    lot = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
    pricef = next(f for f in info["filters"] if f["filterType"] == "PRICE_FILTER")
    notional = next((f for f in info["filters"] if f["filterType"] in ("NOTIONAL","MIN_NOTIONAL")), {})
    min_notional = d(notional.get("minNotional", MIN_NOTIONAL_FALLBACK)) if notional else MIN_NOTIONAL_FALLBACK

    min_qty = d(lot["minQty"])
    step_qty = d(lot["stepSize"])
    tick = d(pricef["tickSize"])

    qty_dp = decimals_from_step(step_qty)
    px_dp  = decimals_from_step(tick)
    return min_qty, step_qty, tick, min_notional, qty_dp, px_dp

def decimals_from_step(step: Decimal) -> int:
    # e.g. step "0.00100000" -> 3
    s = format(step.normalize(), 'f')
    if '.' in s:
        return len(s.split('.')[1])
    return 0

def quantize_to_step(value: Decimal, step: Decimal) -> Decimal:
    # exact Binance-compliant rounding down to step
    if step == 0:
        return value
    return (value // step) * step

def fmt_decimal_for_step(value: Decimal, step: Decimal, dp_hint: int | None = None) -> str:
    # Format respecting exact decimals allowed by step, avoiding extra precision
    dp = dp_hint if dp_hint is not None else decimals_from_step(step)
    q = value.quantize(Decimal(1).scaleb(-dp), rounding=ROUND_DOWN) if dp > 0 else value.quantize(Decimal(1), rounding=ROUND_DOWN)
    s = f"{q:.{dp}f}" if dp > 0 else f"{q:.0f}"
    # strip trailing zeros but keep required places? Binance accepts exact places; keep them
    return s

# ----- balances & sizing
def get_spot_balance(asset: str) -> tuple[Decimal, Decimal, Decimal]:
    try:
        acct = client.get_account()
    except Exception as e:
        log(f"[ACCOUNT ERROR] {e}")
        return Decimal(0), Decimal(0), Decimal(0)
    free = locked = Decimal(0)
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

def compute_qty(entry: Decimal, stop: Decimal, equity: Decimal,
                min_qty: Decimal, step_qty: Decimal, min_notional: Decimal) -> Decimal:
    # risk in quote
    risk_dollars = equity * RISK_PCT
    rpu = max(entry - stop, entry * Decimal("0.002"))  # % floor for stability
    if rpu <= 0:
        return Decimal(0)

    raw_qty = max(risk_dollars / rpu, min_notional / max(entry, Decimal("1e-12")))
    # quantize to step
    qty = quantize_to_step(raw_qty, step_qty)
    if qty < min_qty:
        return Decimal(0)

    # ensure notional >= min_notional after quantization
    notional = qty * entry
    if notional < min_notional:
        # try to bump to meet notional
        needed = (min_notional / entry)
        bumped = quantize_to_step(needed, step_qty)
        if bumped >= min_qty and bumped * entry >= min_notional:
            qty = bumped
        else:
            return Decimal(0)
    return qty

# ----- order placement
def place_market_buy(symbol: str, qty: Decimal) -> dict | None:
    if DRY_RUN:
        log(f"[MARKET BUY dry] {symbol} qty={qty}")
        return {"orderId": "dry"}
    try:
        o = client.create_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=str(qty))
        log(f"[MARKET BUY] {symbol} qty={qty} id={o.get('orderId')}")
        return o
    except Exception as e:
        log(f"[ENTRY ERROR] {symbol}: {e}")
        return None

def place_stop_limit_buy(symbol: str, qty: Decimal, entry: Decimal, tick: Decimal, px_dp: int) -> dict | None:
    stop_px  = quantize_to_step(entry, tick)
    limit_px = quantize_to_step(entry * (Decimal(1) + STOP_LIMIT_OFFSET), tick)

    stop_s  = fmt_decimal_for_step(stop_px, tick, px_dp)
    limit_s = fmt_decimal_for_step(limit_px, tick, px_dp)
    qty_s   = str(qty)  # already quantized to step, string OK

    if DRY_RUN:
        log(f"[STOP-LIMIT BUY dry] {symbol} qty={qty_s} stop={stop_s} limit={limit_s}")
        return {"orderId": "dry"}

    try:
        o = client.create_order(
            symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_STOP_LOSS_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC, quantity=qty_s,
            price=limit_s, stopPrice=stop_s
        )
        log(f"[STOP-LIMIT BUY] {symbol} qty={qty_s} stop={stop_s} limit={limit_s} id={o.get('orderId')}")
        return o
    except Exception as e:
        log(f"[STOP-LIMIT ERROR] {symbol}: {e}")
        return None

def place_tp_limits(symbol: str, qty: Decimal, t1: Decimal, t2: Decimal, tick: Decimal, px_dp: int, step_qty: Decimal) -> None:
    q1 = quantize_to_step(qty * Decimal("0.5"), step_qty)
    q2 = quantize_to_step(qty - q1, step_qty)

    p1 = quantize_to_step(t1, tick)
    p2 = quantize_to_step(t2, tick)

    p1s = fmt_decimal_for_step(p1, tick, px_dp)
    p2s = fmt_decimal_for_step(p2, tick, px_dp)

    if DRY_RUN:
        log(f"[TP dry] {symbol} sell {q1}@{p1s} and {q2}@{p2s}")
        return
    try:
        o1 = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT,
                                 timeInForce=TIME_IN_FORCE_GTC, quantity=str(q1), price=p1s)
        o2 = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT,
                                 timeInForce=TIME_IN_FORCE_GTC, quantity=str(q2), price=p2s)
        log(f"[TP] {symbol} t1 {q1}@{p1s} id={o1.get('orderId')}; t2 {q2}@{p2s} id={o2.get('orderId')}")
    except Exception as e:
        log(f"[TP ERROR] {symbol}: {e}")

# ----- main flow
def main():
    free, locked, total = get_spot_balance(QUOTE_ASSET)
    if FORCE_EQUITY > 0:
        log(f"Balance: OVERRIDE {FORCE_EQUITY} {QUOTE_ASSET} | live free={free}, locked={locked}, total={total}")
    else:
        log(f"Balance: live free={free}, locked={locked}, total={total} {QUOTE_ASSET}")

    S = fetch_summary()
    if not S:
        return

    state = load_state()
    state["_last_run"] = now_str()
    state["_last_balance"] = {"asset": QUOTE_ASSET, "free": float(free), "locked": float(locked), "total": float(total)}

    # Candidates -> stop-limit buy at entry (reduced risk)
    if TRADE_CANDS:
        eq_c = equity_for_sizing() * max(Decimal(0), C_RISK_MULT)
        for c in S.get("candidates", []):
            sym = c["symbol"]
            if not sym.endswith(QUOTE_ASSET): 
                continue
            entry = d(c["entry"]); stop = d(c["stop"]); t1=d(c["t1"]); t2=d(c["t2"])
            try:
                min_qty, step_qty, tick, min_notional, qty_dp, px_dp = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}"); continue

            qty = compute_qty(entry, stop, eq_c, min_qty, step_qty, min_notional)
            if qty <= 0:
                log(f"[CAND SKIP] {sym} qty too small")
                continue

            o = place_stop_limit_buy(sym, qty, entry, tick, px_dp)
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
            sym = o["symbol"]
            if not sym.endswith(QUOTE_ASSET): 
                continue
            entry = d(o["entry"]); stop = d(o["stop"]); t1=d(o["t1"]); t2=d(o["t2"])
            try:
                min_qty, step_qty, tick, min_notional, qty_dp, px_dp = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}"); continue

            qty = compute_qty(entry, stop, eq_b, min_qty, step_qty, min_notional)
            if qty <= 0:
                log(f"[BREAKOUT SKIP] {sym} qty too small")
                continue

            m = place_market_buy(sym, qty)
            if m:
                place_tp_limits(sym, qty, t1, t2, tick, px_dp, step_qty)
                state[sym] = {
                    "entry": float(entry), "stop": float(stop), "t1": float(t1), "t2": float(t2),
                    "filled_qty": float(qty), "t1_filled_qty": 0.0,
                    "status": "live", "entry_order_id": m.get("orderId"),
                    "type": "B"
                }

    save_state(state)

if __name__ == "__main__":
    main()