#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, math, requests
from pathlib import Path
from datetime import datetime, timezone, timedelta

from binance.client import Client
from binance.enums import (
    SIDE_BUY, SIDE_SELL, TIME_IN_FORCE_GTC,
    ORDER_TYPE_MARKET, ORDER_TYPE_STOP_LOSS_LIMIT, ORDER_TYPE_LIMIT
)

# ========= ENV / CONFIG =========
SUMMARY_URL         = os.getenv("SUMMARY_URL", "https://raw.githubusercontent.com/armendq/bincrypare/main/public_runs/latest/summary.json")
QUOTE_ASSET         = os.getenv("QUOTE_ASSET", "USDC")
DRY_RUN             = os.getenv("DRY_RUN", "1") == "1"

FORCE_EQUITY        = float(os.getenv("FORCE_EQUITY_USD", "0"))
RISK_PCT            = float(os.getenv("RISK_PCT", "0.012"))   # 1.2% default

# Candidate trading
TRADE_CANDS         = os.getenv("TRADE_CANDIDATES", "1") == "1"
C_RISK_MULT         = float(os.getenv("C_RISK_MULT", "0.7"))
STOP_LIMIT_OFFSET   = float(os.getenv("STOP_LIMIT_OFFSET", "0.002"))  # 0.2% over stop

# Exchange / filters
MIN_NOTIONAL_FALLBACK = float(os.getenv("MIN_NOTIONAL", "5.0"))
MIN_ORDER_USD       = float(os.getenv("MIN_ORDER_USD", "6.0"))
ALLOW_MIN_ORDER     = os.getenv("ALLOW_MIN_ORDER", "1") == "1"

# Lifecycle rules
PEND_MAX_MINUTES        = int(os.getenv("PEND_MAX_MINUTES", "180"))  # 3h
PROMOTE_MIN_FILL_PCT    = float(os.getenv("PROMOTE_MIN_FILL_PCT", "0.25"))  # promote after 25% filled
LIVE_MAX_HOURS          = float(os.getenv("LIVE_MAX_HOURS", "12"))  # max live age
STOP_SLIPPAGE_PCT       = float(os.getenv("STOP_SLIPPAGE_PCT", "0.001"))  # 0.1% under stop for market exit

API_KEY   = os.getenv("BINANCE_API_KEY", "")
API_SECRET= os.getenv("BINANCE_API_SECRET", "")

STATE_DIR  = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "open_positions.json"

client = Client(API_KEY, API_SECRET)

# ========= UTILS =========
def now_utc():
    return datetime.now(timezone.utc)

def now_str():
    return now_utc().astimezone().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
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

def fetch_summary() -> dict | None:
    # Support local file:// or remote URL
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
            r = requests.get(SUMMARY_URL, timeout=25)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2)
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
    if free < MIN_ORDER_USD:
        log(f"[EQUITY SKIP] free={free:.2f} < MIN_ORDER_USD={MIN_ORDER_USD}")
        return 0.0
    return free

def floor_to_step(v: float, step: float) -> float:
    if step <= 0:
        return v
    return math.floor(v / step) * step

def round_to_tick(p: float, tick: float) -> float:
    if tick <= 0:
        return p
    return floor_to_step(p, tick)

def get_symbol_filters(symbol: str) -> tuple[float, float, float, float]:
    """Return (minQty, stepQty, tickSize, minNotional)"""
    info = client.get_symbol_info(symbol)
    lot = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
    pricef = next(f for f in info["filters"] if f["filterType"] == "PRICE_FILTER")
    notional = next((f for f in info["filters"] if f["filterType"] in ("NOTIONAL","MIN_NOTIONAL")), {})
    min_notional = float(notional.get("minNotional", MIN_NOTIONAL_FALLBACK)) if notional else MIN_NOTIONAL_FALLBACK
    return float(lot["minQty"]), float(lot["stepSize"]), float(pricef["tickSize"]), max(MIN_NOTIONAL_FALLBACK, min_notional)

def get_last_price(symbol: str) -> float | None:
    try:
        t = client.get_symbol_ticker(symbol=symbol)
        return float(t["price"])
    except Exception:
        return None

# ========= SIZING =========
def compute_qty(entry: float, stop: float, equity: float,
                min_qty: float, step_qty: float, min_notional: float,
                allow_min_order: bool) -> float:
    """Risk-based position size with MIN_NOTIONAL guard and optional min-order fallback."""
    if equity <= 0 or entry <= 0 or stop <= 0:
        return 0.0
    risk_dollars = equity * RISK_PCT
    rpu = max(entry - stop, entry * 0.001)  # per-unit risk, floor at 0.1% of entry
    if rpu <= 0:
        return 0.0
    raw_qty = risk_dollars / rpu
    # Ensure notional at entry meets min_notional
    raw_qty = max(raw_qty, min_notional / entry)
    qty = floor_to_step(raw_qty, step_qty)
    if qty < min_qty:
        # Optionally try the absolute minimum order
        if allow_min_order:
            qty = floor_to_step(max(min_qty, min_notional / entry), step_qty)
        else:
            return 0.0
    return qty

# ========= ORDER PLACEMENT =========
def place_market_buy(symbol: str, qty: float) -> dict | None:
    if DRY_RUN:
        log(f"[MARKET BUY dry] {symbol} qty={qty:.8f}")
        return {"orderId": "dry", "executedQty": f"{qty:.8f}", "status": "FILLED"}
    try:
        o = client.create_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=qty)
        log(f"[MARKET BUY] {symbol} qty={qty:.8f} id={o.get('orderId')}")
        return o
    except Exception as e:
        log(f"[ENTRY ERROR] {symbol}: {e}")
        return None

def place_market_sell(symbol: str, qty: float) -> dict | None:
    if qty <= 0:
        return None
    if DRY_RUN:
        log(f"[MARKET SELL dry] {symbol} qty={qty:.8f}")
        return {"orderId":"dry", "executedQty": f"{qty:.8f}", "status": "FILLED"}
    try:
        o = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=qty)
        log(f"[MARKET SELL] {symbol} qty={qty:.8f} id={o.get('orderId')}")
        return o
    except Exception as e:
        log(f"[SELL ERROR] {symbol}: {e}")
        return None

def place_stop_limit_buy(symbol: str, qty: float, entry: float, tick: float,
                         min_notional: float, free_funds: float) -> dict | None:
    stop_px  = round_to_tick(entry, tick)
    limit_px = round_to_tick(entry * (1 + STOP_LIMIT_OFFSET), tick)

    # Pre-checks to avoid 2010/1013
    last_px = get_last_price(symbol)
    if last_px is not None and stop_px <= last_px:
        log(f"[SKIP STOP-LIMIT] {symbol} stop={stop_px} <= lastPrice={last_px}")
        return None

    notional = qty * limit_px
    if notional < min_notional:
        log(f"[STOP-LIMIT SKIP] {symbol} notional {notional:.4f} < minNotional {min_notional:.4f}")
        return None
    if notional > max(0.0, free_funds - 0.01):
        log(f"[STOP-LIMIT SKIP] {symbol} need {notional:.4f} > free {free_funds:.4f}")
        return None

    if DRY_RUN:
        log(f"[STOP-LIMIT BUY dry] {symbol} qty={qty:.8f} stop={stop_px} limit={limit_px}")
        return {"orderId":"dry", "status":"NEW"}

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

def place_tp_limits(symbol: str, qty: float, t1: float, t2: float, tick: float,
                    step_qty: float, min_qty: float, min_notional: float) -> tuple[dict|None, dict|None]:
    """Place two TP limit sells; if either leg violates minNotional/lot, fall back to single TP at t2."""
    if qty <= 0:
        return None, None

    qty = floor_to_step(qty, step_qty)  # refloor executed qty
    if qty < min_qty:
        log(f"[TP SKIP] {symbol} qty {qty:.8f} < minQty {min_qty}")
        return None, None

    q1 = floor_to_step(qty * 0.5, step_qty)
    q2 = floor_to_step(qty - q1, step_qty)
    if q1 < min_qty:  # push all to q2 if q1 too small
        q1 = 0.0
        q2 = floor_to_step(qty, step_qty)

    p1 = round_to_tick(t1, tick)
    p2 = round_to_tick(t2, tick)

    not1 = q1 * p1
    not2 = q2 * p2
    # Fallback to single TP if any leg breaches minNotional
    if (q1 > 0 and not1 < min_notional) or (q2 > 0 and not2 < min_notional):
        if qty * p2 < min_notional:
            log(f"[TP SKIP] {symbol} total notional {qty*p2:.4f} < minNotional {min_notional:.4f}")
            return None, None
        if DRY_RUN:
            log(f"[TP dry fallback] {symbol} sell {qty:.8f}@{p2}")
            return {"orderId":"dry"}, None
        try:
            o = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT,
                                    timeInForce=TIME_IN_FORCE_GTC, quantity=qty, price=f"{p2:.08f}")
            log(f"[TP fallback] {symbol} all {qty:.8f}@{p2} id={o.get('orderId')}")
            return o, None
        except Exception as e:
            log(f"[TP ERROR] {symbol}: {e}")
            return None, None

    if DRY_RUN:
        if q1 > 0: log(f"[TP dry] {symbol} {q1:.8f}@{p1}")
        if q2 > 0: log(f"[TP dry] {symbol} {q2:.8f}@{p2}")
        return ({"orderId":"dry"} if q1>0 else None), ({"orderId":"dry"} if q2>0 else None)

    try:
        o1 = None
        if q1 > 0:
            o1 = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT,
                                     timeInForce=TIME_IN_FORCE_GTC, quantity=q1, price=f"{p1:.08f}")
        o2 = None
        if q2 > 0:
            o2 = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT,
                                     timeInForce=TIME_IN_FORCE_GTC, quantity=q2, price=f"{p2:.08f}")
        if o1: log(f"[TP] {symbol} t1 {q1:.8f}@{p1} id={o1.get('orderId')}")
        if o2: log(f"[TP] {symbol} t2 {q2:.8f}@{p2} id={o2.get('orderId')}")
        return o1, o2
    except Exception as e:
        log(f"[TP ERROR] {symbol}: {e}")
        return None, None

# ========= WORKFLOW HELPERS =========
def ensure_filters_in_pos(symbol: str, pos: dict) -> None:
    """Guarantee tick/step/minQty/minNotional are present in the state slot."""
    for k in ("min_qty","step_qty","tick","min_notional"):
        if k not in pos:
            min_qty, step_qty, tick, min_notional = get_symbol_filters(symbol)
            pos["min_qty"] = min_qty
            pos["step_qty"] = step_qty
            pos["tick"] = tick
            pos["min_notional"] = min_notional
            pos.setdefault("ts", now_utc().timestamp())
            return
    # already present

def age_hours(ts: float) -> float:
    try:
        return max(0.0, (now_utc() - datetime.fromtimestamp(ts, tz=timezone.utc)).total_seconds() / 3600.0)
    except Exception:
        return 0.0

def maybe_activate_pending(symbol: str, pos: dict) -> None:
    """If a pending stop-limit filled, promote to live and place TPs."""
    ensure_filters_in_pos(symbol, pos)
    oid = pos.get("entry_order_id")
    if not oid:
        return
    try:
        o = client.get_order(symbol=symbol, orderId=oid)
    except Exception as e:
        log(f"[ORDER QUERY] {symbol} {e}")
        return

    status = (o.get("status") or "").upper()
    executed = float(o.get("executedQty", "0") or 0)

    # Cancel stale pending
    if status in ("NEW","PARTIALLY_FILLED"):
        # TTL
        if age_hours(float(pos.get("ts", now_utc().timestamp()))) >= (PEND_MAX_MINUTES / 60.0):
            log(f"[PENDING CANCEL TTL] {symbol} > {PEND_MAX_MINUTES}m")
            if not DRY_RUN:
                try: client.cancel_order(symbol=symbol, orderId=oid)
                except Exception as e: log(f"[CANCEL ERR] {symbol} {e}")
            pos["status"] = "canceled"
            return

        # Promote once partial fills exceed threshold
        orig_qty = float(o.get("origQty", "0") or 0)
        if orig_qty > 0 and executed / orig_qty >= PROMOTE_MIN_FILL_PCT:
            pos["filled_qty"] = executed

            # Place TP on executed part
            o1,o2 = place_tp_limits(
                symbol,
                executed,
                pos["t1"], pos["t2"],
                pos["tick"], pos["step_qty"], pos["min_qty"], pos["min_notional"]
            )
            pos["status"] = "live"
            if o1: pos["tp1_order_id"] = o1.get("orderId")
            if o2: pos["tp2_order_id"] = o2.get("orderId")
            log(f"[PROMOTE] {symbol} pending -> live (qty={executed:.8f})")
            return

    if status == "FILLED":
        # Promote full fill
        executed = float(o.get("executedQty", "0") or 0)
        executed = floor_to_step(executed, pos["step_qty"])
        pos["filled_qty"] = executed
        o1,o2 = place_tp_limits(
            symbol, executed,
            pos["t1"], pos["t2"],
            pos["tick"], pos["step_qty"], pos["min_qty"], pos["min_notional"]
        )
        pos["status"] = "live"
        if o1: pos["tp1_order_id"] = o1.get("orderId")
        if o2: pos["tp2_order_id"] = o2.get("orderId")
        log(f"[PROMOTE] {symbol} pending -> live (full)")
        return

    if status in ("CANCELED","REJECTED","EXPIRED"):
        pos["status"] = "canceled"

def manage_live(symbol: str, pos: dict) -> None:
    """Time/stop management for live positions."""
    ensure_filters_in_pos(symbol, pos)
    qty = float(pos.get("filled_qty", 0.0))
    if qty <= 0:
        return

    # Time-based exit
    if age_hours(float(pos.get("ts", now_utc().timestamp()))) >= LIVE_MAX_HOURS:
        log(f"[TIME EXIT] {symbol} live>{LIVE_MAX_HOURS}h")
        if not DRY_RUN:
            qty_s = floor_to_step(qty, pos["step_qty"])
            place_market_sell(symbol, qty_s)
        pos["status"] = "closed"
        return

    # Stop-based exit
    last_px = get_last_price(symbol)
    if last_px is None:
        return
    stop = float(pos.get("stop", 0.0) or 0)
    if stop > 0 and last_px <= stop * (1 - STOP_SLIPPAGE_PCT):
        qty_s = floor_to_step(qty, pos["step_qty"])
        notional = qty_s * last_px
        if qty_s <= 0 or notional < pos["min_notional"]:
            log(f"[STOP EXIT SKIP] {symbol} qty/notional too small ({qty_s}, {notional:.4f})")
            return
        if DRY_RUN:
            log(f"[STOP EXIT dry] {symbol} qty={qty_s:.8f} last={last_px}")
        else:
            place_market_sell(symbol, qty_s)
        pos["status"] = "stopped"

# ========= MAIN =========
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

    # 1) Maintain existing positions
    for sym, pos in list(state.items()):
        if not isinstance(pos, dict) or sym.startswith("_"):
            continue
        try:
            if pos.get("status") == "pending":
                maybe_activate_pending(sym, pos)
            elif pos.get("status") == "live":
                manage_live(sym, pos)
        except Exception as e:
            log(f"[MAINTAIN ERROR] {sym}: {e}")

    # 2) Trade new candidates (stop-limit) and breakouts (market)
    equity = equity_for_sizing()

    # Candidates -> stop-limit buy at entry (reduced risk)
    if TRADE_CANDS and equity > 0:
        free_now, _, _ = get_spot_balance(QUOTE_ASSET)
        for c in S.get("candidates", []):
            sym = c.get("symbol")
            if not sym or not sym.endswith(QUOTE_ASSET):
                continue
            entry = float(c["entry"]); stop = float(c["stop"])
            t1 = float(c["t1"]); t2 = float(c["t2"])

            try:
                min_qty, step_qty, tick, min_notional = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}"); continue

            qty = compute_qty(entry, stop, equity * max(0.0, C_RISK_MULT),
                              min_qty, step_qty, min_notional, ALLOW_MIN_ORDER)
            qty = floor_to_step(qty, step_qty)
            if qty < min_qty:
                log(f"[CAND SKIP] {sym} qty too small")
                continue

            o = place_stop_limit_buy(sym, qty, entry, tick, min_notional, free_now)
            if o:
                state[sym] = {
                    "entry": entry, "stop": stop, "t1": t1, "t2": t2,
                    "filled_qty": 0.0, "t1_filled_qty": 0.0,
                    "status": "pending", "entry_order_id": o.get("orderId"),
                    "type": "C",
                    # cache filters for later TP/rounding
                    "min_qty": min_qty, "step_qty": step_qty, "tick": tick, "min_notional": min_notional,
                    "ts": now_utc().timestamp()
                }

    # Breakouts -> market buy now + place TPs
    if S.get("signals", {}).get("type") == "B" and equity > 0:
        free_now, _, _ = get_spot_balance(QUOTE_ASSET)
        for o in S.get("orders", []) or []:
            sym = o.get("symbol")
            if not sym or not sym.endswith(QUOTE_ASSET):
                continue
            entry = float(o["entry"]); stop = float(o["stop"])
            t1 = float(o["t1"]); t2 = float(o["t2"])

            try:
                min_qty, step_qty, tick, min_notional = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}"); continue

            qty = compute_qty(entry, stop, equity, min_qty, step_qty, min_notional, ALLOW_MIN_ORDER)
            qty = floor_to_step(qty, step_qty)
            notional = qty * entry
            if qty < min_qty or notional < min_notional or notional > free_now:
                log(f"[BREAKOUT SKIP] {sym} qty/notional invalid (qty={qty:.8f}, notional={notional:.4f}, free={free_now:.4f})")
                continue

            m = place_market_buy(sym, qty)
            if m:
                o1, o2 = place_tp_limits(sym, qty, t1, t2, tick, step_qty, min_qty, min_notional)
                state[sym] = {
                    "entry": entry, "stop": stop, "t1": t1, "t2": t2,
                    "filled_qty": qty, "t1_filled_qty": 0.0,
                    "status": "live", "entry_order_id": m.get("orderId"),
                    "type": "B",
                    "min_qty": min_qty, "step_qty": step_qty, "tick": tick, "min_notional": min_notional,
                    "ts": now_utc().timestamp(),
                    "tp1_order_id": (o1 or {}).get("orderId") if o1 else None,
                    "tp2_order_id": (o2 or {}).get("orderId") if o2 else None,
                }

    save_state(state)

if __name__ == "__main__":
    main()