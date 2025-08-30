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

# High precision; always ROUND_DOWN to satisfy Binance filters
getcontext().prec = 28

# ----------------- ENV -----------------
SUMMARY_URL       = os.getenv("SUMMARY_URL", "https://raw.githubusercontent.com/armendq/bincrypare/main/public_runs/latest/summary.json")
QUOTE_ASSET       = os.getenv("QUOTE_ASSET", "USDC")
DRY_RUN           = os.getenv("DRY_RUN", "1") == "1"
FORCE_EQUITY      = Decimal(os.getenv("FORCE_EQUITY_USD", "0"))
TRADE_CANDS       = os.getenv("TRADE_CANDIDATES", "1") == "1"
C_RISK_MULT       = Decimal(os.getenv("C_RISK_MULT", "0.5"))
RISK_PCT          = Decimal(os.getenv("RISK_PCT", "0.012"))
STOP_LIMIT_OFFSET = Decimal(os.getenv("STOP_LIMIT_OFFSET", "0.003"))  # ↑ default 0.3%
MAX_SLIPPAGE      = Decimal(os.getenv("MAX_SLIPPAGE", "0.004"))       # 0.4% market buy tolerance
MIN_NOTIONAL_FALLBACK = Decimal(os.getenv("MIN_NOTIONAL", "5.0"))

API_KEY   = os.getenv("BINANCE_API_KEY", "")
API_SECRET= os.getenv("BINANCE_API_SECRET", "")

STATE_DIR  = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "open_positions.json"

client = Client(API_KEY, API_SECRET)

# ----------------- UTILS -----------------
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

# -------- Binance filters & quantization --------
def get_symbol_filters(symbol: str):
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
    s = format(step.normalize(), 'f')
    return len(s.split('.')[1]) if '.' in s else 0

def quantize_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step == 0:
        return value
    return (value // step) * step  # exact n*step

def fmt_decimal_for_step(value: Decimal, step: Decimal, dp_hint: int | None = None) -> str:
    dp = dp_hint if dp_hint is not None else decimals_from_step(step)
    q = value.quantize(Decimal(1).scaleb(-dp), rounding=ROUND_DOWN) if dp > 0 else value.quantize(Decimal(1), rounding=ROUND_DOWN)
    return f"{q:.{dp}f}" if dp > 0 else f"{q:.0f}"

# -------- balances & sizing --------
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
    risk_dollars = equity * RISK_PCT
    rpu = max(entry - stop, entry * Decimal("0.002"))
    if rpu <= 0:
        return Decimal(0)
    raw_qty = max(risk_dollars / rpu, min_notional / max(entry, Decimal("1e-12")))
    qty = quantize_to_step(raw_qty, step_qty)
    if qty < min_qty:
        return Decimal(0)
    # ensure notional after quantization
    if qty * entry < min_notional:
        bump = quantize_to_step((min_notional / entry), step_qty)
        if bump >= min_qty and bump * entry >= min_notional:
            qty = bump
        else:
            return Decimal(0)
    return qty

# -------- order placement --------
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

    if DRY_RUN:
        log(f"[STOP-LIMIT BUY dry] {symbol} qty={qty} stop={stop_s} limit={limit_s}")
        return {"orderId": "dry"}

    try:
        o = client.create_order(
            symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_STOP_LOSS_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC, quantity=str(qty),
            price=limit_s, stopPrice=stop_s
        )
        log(f"[STOP-LIMIT BUY] {symbol} qty={qty} stop={stop_s} limit={limit_s} id={o.get('orderId')}")
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

# -------- helpers for chaser --------
def get_last_price(symbol: str) -> Decimal | None:
    try:
        t = client.get_symbol_ticker(symbol=symbol)
        return d(t["price"])
    except Exception as e:
        log(f"[PRICE ERROR] {symbol}: {e}")
        return None

def cancel_order(symbol: str, order_id: str | int) -> None:
    if DRY_RUN:
        log(f"[CANCEL dry] {symbol} orderId={order_id}")
        return
    try:
        client.cancel_order(symbol=symbol, orderId=order_id)
        log(f"[CANCEL] {symbol} orderId={order_id}")
    except Exception as e:
        log(f"[CANCEL ERROR] {symbol}: {e}")

# ----------------- MAIN -----------------
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

    # -------- CHASER: convert pending stop-limits to market if crossed mildly --------
    for sym, pos in list(state.items()):
        if not isinstance(pos, dict): 
            continue
        if pos.get("status") != "pending":
            continue
        if not sym.endswith(QUOTE_ASSET):
            continue

        stop = d(pos["stop"])
        entry_order_id = pos.get("entry_order_id")
        last = get_last_price(sym)
        if last is None:
            continue

        if last >= stop and (last / stop - Decimal(1)) <= MAX_SLIPPAGE:
            log(f"[CHASE] {sym} last={last} >= stop={stop} within {MAX_SLIPPAGE*100:.2f}% → market buy")
            # best effort cancel existing stop-limit
            if entry_order_id:
                cancel_order(sym, entry_order_id)

            # recompute qty from stored entry/stop with live equity
            eq = equity_for_sizing()
            try:
                min_qty, step_qty, tick, min_notional, qty_dp, px_dp = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}"); continue

            entry = d(pos["entry"])
            stopd = d(pos["stop"])
            qty = compute_qty(entry, stopd, eq, min_qty, step_qty, min_notional)
            if qty <= 0:
                log(f"[CHASE SKIP] {sym} qty too small after recompute")
                continue

            m = place_market_buy(sym, qty)
            if m:
                place_tp_limits(sym, qty, d(pos["t1"]), d(pos["t2"]), tick, px_dp, step_qty)
                pos.update({
                    "filled_qty": float(qty),
                    "status": "live",
                    "entry_order_id": m.get("orderId"),
                    "type": pos.get("type", "C")
                })
                state[sym] = pos

    # -------- NEW CANDIDATES FROM LATEST SCAN --------
    if TRADE_CANDS:
        eq_c = equity_for_sizing() * max(Decimal(0), C_RISK_MULT)
        for c in S.get("candidates", []):
            sym = c["symbol"]
            if not sym.endswith(QUOTE_ASSET): 
                continue
            if sym in state and state[sym].get("status") in ("pending","live"):
                continue  # already working
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

    # -------- INSTANT BREAKOUTS --------
    if S.get("signals", {}).get("type") == "B":
        eq_b = equity_for_sizing()
        for o in S.get("orders", []) or []:
            sym = o["symbol"]
            if not sym.endswith(QUOTE_ASSET): 
                continue
            if sym in state and state[sym].get("status") == "live":
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