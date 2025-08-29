#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, math, requests
from pathlib import Path
from datetime import datetime, timezone
from binance.client import Client
from binance.enums import (
    SIDE_BUY, SIDE_SELL, TIME_IN_FORCE_GTC,
    ORDER_TYPE_MARKET, ORDER_TYPE_STOP_LOSS_LIMIT, ORDER_TYPE_LIMIT
)

# env
SUMMARY_URL       = os.getenv("SUMMARY_URL", "https://raw.githubusercontent.com/armendq/bincrypare/main/public_runs/latest/summary.json")
QUOTE_ASSET       = os.getenv("QUOTE_ASSET", "USDC")
DRY_RUN           = os.getenv("DRY_RUN", "1") == "1"
FORCE_EQUITY      = float(os.getenv("FORCE_EQUITY_USD", "0"))
TRADE_CANDS       = os.getenv("TRADE_CANDIDATES", "1") == "1"
C_RISK_MULT       = float(os.getenv("C_RISK_MULT", "0.5"))
RISK_PCT          = float(os.getenv("RISK_PCT", "0.012"))
STOP_LIMIT_OFFSET = float(os.getenv("STOP_LIMIT_OFFSET", "0.001"))
MIN_NOTIONAL_FALLBACK = float(os.getenv("MIN_NOTIONAL", "5.0"))

API_KEY   = os.getenv("BINANCE_API_KEY", "")
API_SECRET= os.getenv("BINANCE_API_SECRET", "")

STATE_DIR  = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "open_positions.json"

client = Client(API_KEY, API_SECRET)

def now_str() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str) -> None:
    print(f"[exec {now_str()}] {msg}", flush=True)

def load_state() -> dict:
    if STATE_FILE.exists():
        try: return json.loads(STATE_FILE.read_text("utf-8"))
        except Exception: return {}
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
        time.sleep(60)
    log("Hold and wait. Fetch failed twice.")
    return None

def get_spot_balance(asset: str) -> tuple[float, float, float]:
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
    if FORCE_EQUITY > 0:
        return FORCE_EQUITY
    free, _, _ = get_spot_balance(QUOTE_ASSET)
    return free

def get_symbol_filters(symbol: str) -> tuple[float, float, float, float]:
    info = client.get_symbol_info(symbol)
    lot = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
    pricef = next(f for f in info["filters"] if f["filterType"] == "PRICE_FILTER")
    notional = next((f for f in info["filters"] if f["filterType"] in ("NOTIONAL","MIN_NOTIONAL")), {})
    min_notional = float(notional.get("minNotional", MIN_NOTIONAL_FALLBACK)) if notional else MIN_NOTIONAL_FALLBACK
    return float(lot["minQty"]), float(lot["stepSize"]), float(pricef["tickSize"]), max(MIN_NOTIONAL_FALLBACK, min_notional)

def floor_to_step(v: float, step: float) -> float:
    if step <= 0: return v
    return math.floor(v / step) * step

def round_to_tick(p: float, tick: float) -> float:
    if tick <= 0: return p
    return floor_to_step(p, tick)

def compute_qty(entry: float, stop: float, equity: float, min_qty: float, step_qty: float, min_notional: float) -> float:
    risk_dollars = equity * RISK_PCT
    rpu = max(entry - stop, entry * 0.002)
    if rpu <= 0: return 0.0
    raw_qty = max(risk_dollars / rpu, min_notional / max(entry, 1e-12))
    qty = floor_to_step(raw_qty, step_qty)
    if qty < min_qty: return 0.0
    return qty

def place_market_buy(symbol: str, qty: float) -> dict | None:
    if DRY_RUN:
        log(f"[MARKET BUY dry] {symbol} qty={qty:.8f}")
        return {"orderId":"dry"}
    try:
        o = client.create_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=qty)
        log(f"[MARKET BUY] {symbol} qty={qty:.8f} id={o.get('orderId')}")
        return o
    except Exception as e:
        log(f"[ENTRY ERROR] {symbol}: {e}")
        return None

def place_stop_limit_buy(symbol: str, qty: float, entry: float, tick: float) -> dict | None:
    stop_px  = round_to_tick(entry, tick)
    limit_px = round_to_tick(entry * (1 + STOP_LIMIT_OFFSET), tick)
    if DRY_RUN:
        log(f"[STOP-LIMIT BUY dry] {symbol} qty={qty:.8f} stop={stop_px} limit={limit_px}")
        return {"orderId":"dry"}
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

def place_tp_limits(symbol: str, qty: float, t1: float, t2: float, tick: float) -> None:
    q1 = floor_to_step(qty * 0.5, 1e-8)
    q2 = max(qty - q1, 0.0)
    p1 = round_to_tick(t1, tick)
    p2 = round_to_tick(t2, tick)
    if DRY_RUN:
        log(f"[TP dry] {symbol} sell {q1:.8f}@{p1} and {q2:.8f}@{p2}")
        return
    try:
        o1 = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT,
                                 timeInForce=TIME_IN_FORCE_GTC, quantity=q1, price=f"{p1:.08f}")
        o2 = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT,
                                 timeInForce=TIME_IN_FORCE_GTC, quantity=q2, price=f"{p2:.08f}")
        log(f"[TP] {symbol} t1 {q1:.8f}@{p1} id={o1.get('orderId')}; t2 {q2:.8f}@{p2} id={o2.get('orderId')}")
    except Exception as e:
        log(f"[TP ERROR] {symbol}: {e}")

def main():
    free, locked, total = get_spot_balance(QUOTE_ASSET)
    if FORCE_EQUITY > 0:
        log(f"Balance: OVERRIDE {FORCE_EQUITY:.2f} {QUOTE_ASSET} | live free={free:.2f}, locked={locked:.2f}, total={total:.2f}")
    else:
        log(f"Balance: live free={free:.2f}, locked={locked:.2f}, total={total:.2f} {QUOTE_ASSET}")

    S = fetch_summary()
    if not S:
        return

    state = load_state()
    state["_last_run"] = now_str()
    state["_last_balance"] = {"asset": QUOTE_ASSET, "free": round(free,8), "locked": round(locked,8), "total": round(total,8)}

    # Candidates -> stop-limit buy at entry (reduced risk)
    if TRADE_CANDS:
        eq_c = equity_for_sizing() * max(0.0, C_RISK_MULT)
        for c in S.get("candidates", []):
            sym = c["symbol"]
            if not sym.endswith(QUOTE_ASSET): continue
            entry = float(c["entry"]); stop = float(c["stop"]); t1=float(c["t1"]); t2=float(c["t2"])
            try:
                min_qty, step_qty, tick, min_notional = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}"); continue
            qty = compute_qty(entry, stop, eq_c, min_qty, step_qty, min_notional)
            if qty <= 0:
                log(f"[CAND SKIP] {sym} qty too small")
                continue
            o = place_stop_limit_buy(sym, qty, entry, tick)
            if o:
                state[sym] = {
                    "entry": entry, "stop": stop, "t1": t1, "t2": t2,
                    "filled_qty": 0.0, "t1_filled_qty": 0.0,
                    "status": "pending", "entry_order_id": o.get("orderId"),
                    "type": "C"
                }

    # Breakouts -> market buy now + place TPs
    if S.get("signals", {}).get("type") == "B":
        eq_b = equity_for_sizing()
        for o in S.get("orders", []) or []:
            sym = o["symbol"]
            if not sym.endswith(QUOTE_ASSET): continue
            entry = float(o["entry"]); stop = float(o["stop"]); t1=float(o["t1"]); t2=float(o["t2"])
            try:
                min_qty, step_qty, tick, min_notional = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}"); continue
            qty = compute_qty(entry, stop, eq_b, min_qty, step_qty, min_notional)
            if qty <= 0:
                log(f"[BREAKOUT SKIP] {sym} qty too small")
                continue
            m = place_market_buy(sym, qty)
            if m:
                place_tp_limits(sym, qty, t1, t2, tick)
                state[sym] = {
                    "entry": entry, "stop": stop, "t1": t1, "t2": t2,
                    "filled_qty": qty, "t1_filled_qty": 0.0,
                    "status": "live", "entry_order_id": m.get("orderId"),
                    "type": "B"
                }

    save_state(state)

if __name__ == "__main__":
    main()
