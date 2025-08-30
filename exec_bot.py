#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, math
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
from binance.client import Client
from binance.enums import (
    SIDE_BUY, SIDE_SELL, TIME_IN_FORCE_GTC,
    ORDER_TYPE_MARKET, ORDER_TYPE_STOP_LOSS_LIMIT, ORDER_TYPE_LIMIT
)

# -------- Env --------
SUMMARY_URL         = os.getenv("SUMMARY_URL", "https://raw.githubusercontent.com/armendq/bincrypare/main/public_runs/latest/summary.json")
QUOTE_ASSET         = os.getenv("QUOTE_ASSET", "USDC")
DRY_RUN             = os.getenv("DRY_RUN", "1") == "1"
FORCE_EQUITY        = float(os.getenv("FORCE_EQUITY_USD", "0"))
TRADE_CANDS         = os.getenv("TRADE_CANDIDATES", "1") == "1"
C_RISK_MULT         = float(os.getenv("C_RISK_MULT", "0.5"))
RISK_PCT            = float(os.getenv("RISK_PCT", "0.012"))
STOP_LIMIT_OFFSET   = float(os.getenv("STOP_LIMIT_OFFSET", "0.001"))
MIN_NOTIONAL_FALLBACK = float(os.getenv("MIN_NOTIONAL", "5.0"))
ALLOW_MIN_ORDER     = os.getenv("ALLOW_MIN_ORDER", "1") == "1"
MIN_ORDER_USD       = float(os.getenv("MIN_ORDER_USD", "6"))

# Time/risk rules
PENDING_TTL_H       = float(os.getenv("PENDING_TTL_H", "3"))
EARLY_CANCEL_R      = float(os.getenv("EARLY_CANCEL_R", "0.5"))   # cancel pending if last <= entry - 0.5R
LIVE_TTL_H          = float(os.getenv("LIVE_TTL_H", "36"))
STALE_CHECK_H       = float(os.getenv("STALE_CHECK_H", "8"))
STALE_MIN_MFE_R     = float(os.getenv("STALE_MIN_MFE_R", "0.3"))  # close if MFE < 0.3R by STALE_CHECK_H

API_KEY   = os.getenv("BINANCE_API_KEY", "")
API_SECRET= os.getenv("BINANCE_API_SECRET", "")

STATE_DIR  = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "open_positions.json"

client = Client(API_KEY, API_SECRET)

# -------- Utils --------
def now_utc():
    return datetime.now(timezone.utc)

def now_str():
    return now_utc().astimezone().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"[exec {now_str()}] {msg}", flush=True)

def load_state() -> dict:
    if STATE_FILE.exists():
        try: return json.loads(STATE_FILE.read_text("utf-8"))
        except Exception: return {}
    return {}

def save_state(state: dict):
    tmp = STATE_DIR / "open_positions.tmp"
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), "utf-8")
    tmp.replace(STATE_FILE)

def fetch_summary() -> dict | None:
    # local file support
    if SUMMARY_URL.startswith("file://"):
        try:
            p = SUMMARY_URL.replace("file://", "", 1)
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log(f"[LOCAL READ ERROR] {e}")
            return None
    # remote with up to 2 tries
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

def get_spot_balance(asset: str) -> tuple[float,float,float]:
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
    # avoid float drift
    q = math.floor(p / tick + 1e-12)
    return q * tick

def compute_qty(entry: float, stop: float, equity: float, min_qty: float, step_qty: float, min_notional: float) -> float:
    risk_dollars = equity * RISK_PCT
    rpu = max(entry - stop, entry * 0.002)
    if rpu <= 0: return 0.0
    raw_qty = max(risk_dollars / rpu, min_notional / max(entry, 1e-12))
    qty = floor_to_step(raw_qty, step_qty)
    if qty < min_qty: return 0.0
    return qty

def last_price(symbol: str) -> float:
    try:
        t = client.get_symbol_ticker(symbol=symbol)
        return float(t["price"])
    except Exception as e:
        log(f"[TICKER ERROR] {symbol} {e}")
        return 0.0

def order_status(symbol: str, order_id: int | str) -> dict | None:
    try:
        return client.get_order(symbol=symbol, orderId=order_id)
    except Exception as e:
        log(f"[GET ORDER ERROR] {symbol} {order_id} {e}")
        return None

def cancel_order(symbol: str, order_id: int | str) -> bool:
    if DRY_RUN:
        log(f"[CANCEL dry] {symbol} id={order_id}")
        return True
    try:
        client.cancel_order(symbol=symbol, orderId=order_id)
        log(f"[CANCEL] {symbol} id={order_id}")
        return True
    except Exception as e:
        log(f"[CANCEL ERROR] {symbol} {order_id} {e}")
        return False

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

def place_stop_limit_buy(symbol: str, qty: float, entry: float, tick: float) -> dict | None:
    stop_px  = round_to_tick(entry, tick)
    limit_px = round_to_tick(entry * (1 + STOP_LIMIT_OFFSET), tick)
    if DRY_RUN:
        log(f"[STOP-LIMIT BUY dry] {symbol} qty={qty:.8f} stop={stop_px} limit={limit_px}")
        return {"orderId":"dry", "status": "NEW"}
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

def place_tp_limits(symbol: str, qty: float, t1: float, t2: float, tick: float) -> tuple[dict|None, dict|None]:
    q1 = floor_to_step(qty * 0.5, 1e-8)
    q2 = max(qty - q1, 0.0)
    p1 = round_to_tick(t1, tick)
    p2 = round_to_tick(t2, tick)
    if DRY_RUN:
        log(f"[TP dry] {symbol} sell {q1:.8f}@{p1} and {q2:.8f}@{p2}")
        return {"orderId":"dry-t1","status":"NEW"}, {"orderId":"dry-t2","status":"NEW"}
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

def market_sell_all(symbol: str) -> bool:
    # sell total position balance of base asset
    try:
        base = symbol.replace(QUOTE_ASSET, "")
        # find free+locked of base
        acct = client.get_account()
        bal = next((b for b in acct.get("balances", []) if b.get("asset") == base), None)
        if not bal:
            log(f"[SELL] no balance for {base}")
            return False
        free = float(bal.get("free","0")); locked = float(bal.get("locked","0"))
        qty = free  # safest: free only
        if qty <= 0:
            log(f"[SELL] zero free qty for {base}")
            return False
        # round to LOT_SIZE step
        min_qty, step_qty, _, _ = get_symbol_filters(symbol)
        qty = floor_to_step(qty, step_qty)
        if qty < min_qty:
            log(f"[SELL] qty below min lot {qty}<{min_qty}")
            return False
        if DRY_RUN:
            log(f"[MARKET SELL dry] {symbol} qty={qty:.8f}")
            return True
        o = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=qty)
        log(f"[MARKET SELL] {symbol} qty={qty:.8f} id={o.get('orderId')}")
        return True
    except Exception as e:
        log(f"[SELL ERROR] {symbol}: {e}")
        return False

# -------- Core --------
def size_and_submit_candidate(c: dict, equity_avail: float, state: dict):
    sym = c["symbol"]
    if not sym.endswith(QUOTE_ASSET): return
    entry = float(c["entry"]); stop = float(c["stop"]); t1=float(c["t1"]); t2=float(c["t2"])
    try:
        min_qty, step_qty, tick, min_notional = get_symbol_filters(sym)
    except Exception as e:
        log(f"[FILTER] {sym} {e}"); return
    eq = max(0.0, equity_avail * max(0.0, C_RISK_MULT))
    qty = compute_qty(entry, stop, eq, min_qty, step_qty, min_notional)
    if qty <= 0 and ALLOW_MIN_ORDER:
        # fallback: try small notional if allowed
        qty_try = floor_to_step(max(MIN_ORDER_USD, min_notional) / max(entry,1e-12), step_qty)
        if qty_try >= min_qty: qty = qty_try
    if qty <= 0:
        log(f"[CAND SKIP] {sym} qty too small")
        return
    o = place_stop_limit_buy(sym, qty, entry, tick)
    if not o: return
    state[sym] = {
        "entry": entry, "stop": stop, "t1": t1, "t2": t2,
        "tick": tick, "min_qty": min_qty, "step_qty": step_qty,
        "filled_qty": 0.0, "t1_filled_qty": 0.0,
        "status": "pending", "entry_order_id": o.get("orderId"),
        "type": "C", "created_at": now_utc().timestamp()
    }

def submit_breakout(o: dict, equity_avail: float, state: dict):
    sym = o["symbol"]
    if not sym.endswith(QUOTE_ASSET): return
    entry = float(o["entry"]); stop = float(o["stop"]); t1=float(o["t1"]); t2=float(o["t2"])
    try:
        min_qty, step_qty, tick, min_notional = get_symbol_filters(sym)
    except Exception as e:
        log(f"[FILTER] {sym} {e}"); return
    qty = compute_qty(entry, stop, equity_avail, min_qty, step_qty, min_notional)
    if qty <= 0 and ALLOW_MIN_ORDER:
        qty_try = floor_to_step(max(MIN_ORDER_USD, min_notional) / max(entry,1e-12), step_qty)
        if qty_try >= min_qty: qty = qty_try
    if qty <= 0:
        log(f"[B SKIP] {sym} qty too small")
        return
    m = place_market_buy(sym, qty)
    if not m: return
    tp1, tp2 = place_tp_limits(sym, qty, t1, t2, tick)
    state[sym] = {
        "entry": entry, "stop": stop, "t1": t1, "t2": t2,
        "tick": tick, "min_qty": min_qty, "step_qty": step_qty,
        "filled_qty": qty, "t1_filled_qty": 0.0,
        "status": "live", "entry_order_id": m.get("orderId"),
        "tp1_id": tp1.get("orderId") if tp1 else None,
        "tp2_id": tp2.get("orderId") if tp2 else None,
        "type": "B", "filled_at": now_utc().timestamp()
    }

def maybe_activate_pending(sym: str, pos: dict):
    # Check if pending stop-limit buy is filled
    oid = pos.get("entry_order_id")
    if not oid: return
    st = order_status(sym, oid)
    if not st: return
    status = st.get("status")
    if status == "FILLED" or DRY_RUN:
        qty = float(st.get("executedQty", pos.get("filled_qty", 0.0))) if not DRY_RUN else pos.get("filled_qty", 0.0) or float(pos.get("min_qty", 0.0))
        pos["filled_qty"] = qty if qty > 0 else pos.get("filled_qty", 0.0)
        tp1, tp2 = place_tp_limits(sym, pos["filled_qty"], pos["t1"], pos["t2"], pos["tick"])
        pos["tp1_id"] = tp1.get("orderId") if tp1 else None
        pos["tp2_id"] = tp2.get("orderId") if tp2 else None
        pos["status"] = "live"
        pos["filled_at"] = now_utc().timestamp()
        log(f"[ACTIVATED] {sym} -> live qty={pos['filled_qty']:.8f}")
    elif status in ("CANCELED","REJECTED","EXPIRED"):
        pos["status"] = "canceled"
        log(f"[PENDING -> CANCELED] {sym} {status}")

def manage_tp1(sym: str, pos: dict):
    # If TP1 filled, move stop to breakeven for remainder
    t1_id = pos.get("tp1_id")
    if not t1_id: return
    st = order_status(sym, t1_id)
    if not st: return
    if st.get("status") == "FILLED" and pos.get("t1_filled_qty", 0.0) == 0.0:
        executed = float(st.get("executedQty","0") or "0")
        pos["t1_filled_qty"] = executed if executed > 0 else pos["filled_qty"] * 0.5
        # move logical stop to entry (we track logically, executor sells on breach)
        pos["stop"] = max(pos["stop"], pos["entry"])
        log(f"[TP1 HIT] {sym} stop -> breakeven {pos['stop']:.8f}")

def close_if_stop(sym: str, pos: dict, last: float):
    if last <= pos["stop"]:
        ok = market_sell_all(sym)
        pos["status"] = "closed" if ok else pos["status"]
        if ok: log(f"[STOP OUT] {sym} at ~{last}")

def close_if_time(sym: str, pos: dict):
    if pos.get("filled_at"):
        age_h = (now_utc().timestamp() - float(pos["filled_at"])) / 3600.0
        if age_h >= LIVE_TTL_H and pos.get("status") == "live":
            ok = market_sell_all(sym)
            pos["status"] = "closed" if ok else pos["status"]
            if ok: log(f"[TIME STOP] {sym} age={age_h:.1f}h")

def close_if_stale(sym: str, pos: dict, last: float):
    # if after STALE_CHECK_H, price < entry + 0.3R -> close
    if pos.get("filled_at"):
        age_h = (now_utc().timestamp() - float(pos["filled_at"])) / 3600.0
        if age_h >= STALE_CHECK_H and pos.get("status") == "live":
            R = max(pos["entry"] - pos["stop"], 1e-12)
            thresh = pos["entry"] + STALE_MIN_MFE_R * R
            if last < thresh:
                ok = market_sell_all(sym)
                pos["status"] = "closed" if ok else pos["status"]
                if ok: log(f"[STALE EXIT] {sym} last={last:.8f} < {thresh:.8f}")

def cancel_if_expired_or_adverse(sym: str, pos: dict, last: float):
    if pos.get("status") != "pending": return
    created = float(pos.get("created_at", now_utc().timestamp()))
    age_h = (now_utc().timestamp() - created) / 3600.0
    R = max(pos["entry"] - pos["stop"], 1e-12)
    if age_h >= PENDING_TTL_H:
        cancel_order(sym, pos.get("entry_order_id"))
        pos["status"] = "canceled"
        log(f"[PENDING TTL] {sym} canceled after {age_h:.2f}h")
        return
    # early cancel if adverse move −EARLY_CANCEL_R * R
    if last <= pos["entry"] - EARLY_CANCEL_R * R:
        cancel_order(sym, pos.get("entry_order_id"))
        pos["status"] = "canceled"
        log(f"[PENDING ADVERSE] {sym} canceled (last {last:.8f} <= entry - {EARLY_CANCEL_R:.2f}R)")

def all_tp_filled_or_closed(sym: str, pos: dict) -> bool:
    # if both TP orders gone/filled or status closed
    if pos.get("status") == "closed":
        return True
    tp1, tp2 = pos.get("tp1_id"), pos.get("tp2_id")
    done = 0
    for oid in (tp1, tp2):
        if not oid: done += 1; continue
        st = order_status(sym, oid)
        if st and st.get("status") == "FILLED":
            done += 1
    return done == 2

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

    # Place new orders
    if TRADE_CANDS:
        eq_c = equity_for_sizing()
        for c in S.get("candidates", []):
            sym = c["symbol"]
            if sym in state and state[sym].get("status") in ("pending","live"):
                continue
            size_and_submit_candidate(c, eq_c, state)

    if S.get("signals", {}).get("type") == "B":
        eq_b = equity_for_sizing()
        for o in S.get("orders", []) or []:
            sym = o["symbol"]
            if sym in state and state[sym].get("status") in ("pending","live"):
                continue
            submit_breakout(o, eq_b, state)

    # Manage existing
    symbols = [k for k in state.keys() if not k.startswith("_")]
    for sym in symbols:
        pos = state.get(sym, {})
        if not isinstance(pos, dict): continue
        if pos.get("status") in ("canceled","closed"): continue

        lp = last_price(sym)
        if lp <= 0:
            log(f"[SKIP MANAGE] {sym} no last price"); continue

        if pos.get("status") == "pending":
            maybe_activate_pending(sym, pos)
            # still pending? check TTL and adverse move
            if pos.get("status") == "pending":
                cancel_if_expired_or_adverse(sym, pos, lp)
            continue

        if pos.get("status") == "live":
            # Stop check
            close_if_stop(sym, pos, lp)
            if pos.get("status") == "closed": continue
            # TP1 management
            manage_tp1(sym, pos)
            # stale exit
            close_if_stale(sym, pos, lp)
            if pos.get("status") == "closed": continue
            # time stop
            close_if_time(sym, pos)
            if pos.get("status") == "closed": continue
            # if both TP filled, mark closed (do not force market-sell)
            if all_tp_filled_or_closed(sym, pos):
                pos["status"] = "closed"
                log(f"[CLOSED] {sym} both TPs filled or cleared")

    save_state(state)
    log("State saved.")

if __name__ == "__main__":
    main()