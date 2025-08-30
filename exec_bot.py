#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exec_bot.py  —  USDC spot executor

Features
- Places stop-limit BUYs for 'C' candidates (reduced risk sizing).
- Promotes pending -> live when filled (or partial >= PROMOTE_MIN_FILL_PCT).
- Exits:
    * Preferred: TWO OCOs (q1→T1, q2→T2) if both legs satisfy filters.
    * Fallback A: ONE OCO using T2 (entire qty).
    * Fallback B: TP LIMIT (T2) + virtual stop (market sell) if OCO not possible.
- Precision/filters guarded (LOT_SIZE, PRICE_FILTER, NOTIONAL).
- Small-account helpers (ALLOW_MIN_ORDER/MIN_ORDER_USD).
- Resilient against missing cached filters (auto-repair).
"""

import os, json, time, math, requests, traceback
from pathlib import Path
from datetime import datetime, timezone
from binance.client import Client
from binance.enums import (
    SIDE_BUY, SIDE_SELL, TIME_IN_FORCE_GTC,
    ORDER_TYPE_MARKET, ORDER_TYPE_STOP_LOSS_LIMIT, ORDER_TYPE_LIMIT
)

# -------- ENV --------
SUMMARY_URL         = os.getenv("SUMMARY_URL", "https://raw.githubusercontent.com/armendq/bincrypare/main/public_runs/latest/summary.json")
QUOTE_ASSET         = os.getenv("QUOTE_ASSET", "USDC")
DRY_RUN             = os.getenv("DRY_RUN", "1") == "1"
FORCE_EQUITY        = float(os.getenv("FORCE_EQUITY_USD", "0"))
TRADE_CANDS         = os.getenv("TRADE_CANDIDATES", "1") == "1"
C_RISK_MULT         = float(os.getenv("C_RISK_MULT", "0.6"))
RISK_PCT            = float(os.getenv("RISK_PCT", "0.012"))
STOP_LIMIT_OFFSET   = float(os.getenv("STOP_LIMIT_OFFSET", "0.001"))       # 0.1% above entry for stop-limit BUY
PROMOTE_MIN_FILL_PCT= float(os.getenv("PROMOTE_MIN_FILL_PCT", "0.60"))      # 60% partial fill -> live
PRICE_SLIPPAGE      = float(os.getenv("PRICE_SLIPPAGE", "0.003"))           # virtual stop mkt-sell cushion

# Two-OCO split settings
SPLIT_USE_TWO_OCO   = os.getenv("SPLIT_USE_TWO_OCO", "1") == "1"            # try two OCOs if size allows
SPLIT_Q1_FRACTION   = float(os.getenv("SPLIT_Q1_FRACTION", "0.5"))          # q1 for T1, q2 = rest for T2

# Small order helpers
ALLOW_MIN_ORDER     = os.getenv("ALLOW_MIN_ORDER", "1") == "1"
MIN_ORDER_USD       = float(os.getenv("MIN_ORDER_USD", "6.0"))

API_KEY   = os.getenv("BINANCE_API_KEY", "")
API_SECRET= os.getenv("BINANCE_API_SECRET", "")

STATE_DIR  = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "open_positions.json"

client = Client(API_KEY, API_SECRET)

# -------- Utils --------
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
    for _ in (1, 2):
        try:
            r = requests.get(SUMMARY_URL, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(10)
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

def floor_to_step(v: float, step: float) -> float:
    if step <= 0: return v
    return math.floor(v / step) * step

def round_to_tick(p: float, tick: float) -> float:
    if tick <= 0: return p
    return floor_to_step(p, tick)

def get_symbol_filters(symbol: str) -> dict:
    """Return {min_qty, step_qty, tick, min_notional}."""
    info = client.get_symbol_info(symbol)
    lot = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
    pricef = next(f for f in info["filters"] if f["filterType"] == "PRICE_FILTER")
    notional = next((f for f in info["filters"] if f["filterType"] in ("NOTIONAL","MIN_NOTIONAL")), {})
    min_notional = float(notional.get("minNotional", MIN_ORDER_USD)) if notional else MIN_ORDER_USD
    return {
        "min_qty": float(lot["minQty"]),
        "step_qty": float(lot["stepSize"]),
        "tick": float(pricef["tickSize"]),
        "min_notional": max(MIN_ORDER_USD, min_notional)
    }

def compute_qty(entry: float, stop: float, equity: float,
                min_qty: float, step_qty: float, min_notional: float) -> float:
    risk_dollars = equity * RISK_PCT
    rpu = max(entry - stop, entry * 0.002)  # floor risk-per-unit
    if rpu <= 0: return 0.0
    raw_qty = max(risk_dollars / rpu, min_notional / max(entry, 1e-12))
    qty = floor_to_step(raw_qty, step_qty)
    if qty < min_qty: return 0.0
    return qty

# -------- Placers --------
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

def place_limit_sell(symbol: str, qty: float, price: float) -> dict | None:
    if DRY_RUN:
        log(f"[LIMIT SELL dry] {symbol} {qty:.8f}@{price}")
        return {"orderId":"dry-tp"}
    try:
        o = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_LIMIT,
                                timeInForce=TIME_IN_FORCE_GTC, quantity=qty, price=f"{price:.08f}")
        log(f"[LIMIT SELL] {symbol} {qty:.8f}@{price} id={o.get('orderId')}")
        return o
    except Exception as e:
        log(f"[TP ERROR] {symbol}: {e}")
        return None

def place_oco_sell(symbol: str, qty: float, tp_price: float, stop_trig: float, stop_limit: float) -> dict | None:
    if DRY_RUN:
        log(f"[OCO dry] {symbol} qty={qty:.8f} TP={tp_price} stop={stop_trig}/{stop_limit}")
        return {"orderListId":"dry-oco","orders":[{"orderId":"dry-tp"},{"orderId":"dry-sl"}]}
    try:
        o = client.create_oco_order(
            symbol=symbol, side=SIDE_SELL, quantity=qty,
            price=f"{tp_price:.08f}",
            stopPrice=f"{stop_trig:.08f}",
            stopLimitPrice=f"{stop_limit:.08f}",
            stopLimitTimeInForce=TIME_IN_FORCE_GTC
        )
        log(f"[OCO] {symbol} qty={qty:.8f} TP={tp_price} stop={stop_trig}/{stop_limit} listId={o.get('orderListId')}")
        return o
    except Exception as e:
        log(f"[OCO ERROR] {symbol}: {e}")
        return None

def sell_market(symbol: str, qty: float) -> dict | None:
    if DRY_RUN:
        log(f"[MARKET SELL dry] {symbol} {qty:.8f}")
        return {"orderId":"dry-sell"}
    try:
        o = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=qty)
        log(f"[MARKET SELL] {symbol} {qty:.8f} id={o.get('orderId')}")
        return o
    except Exception as e:
        log(f"[MARKET SELL ERROR] {symbol}: {e}")
        return None

# -------- Helpers --------
def symbol_last_price(symbol: str) -> float:
    try:
        t = client.get_symbol_ticker(symbol=symbol)
        return float(t["price"])
    except Exception:
        return 0.0

def order_status(symbol: str, order_id) -> str:
    try:
        o = client.get_order(symbol=symbol, orderId=order_id)
        return o.get("status", "UNKNOWN")
    except Exception:
        return "UNKNOWN"

def repair_filters(sym: str, pos: dict) -> dict:
    """Ensure tick/step/mins exist on the position."""
    missing = any(k not in pos for k in ("tick","step_qty","min_qty","min_notional"))
    if not missing:
        return pos
    try:
        f = get_symbol_filters(sym)
        pos["tick"], pos["step_qty"], pos["min_qty"], pos["min_notional"] = f["tick"], f["step_qty"], f["min_qty"], f["min_notional"]
    except Exception:
        pass
    return pos

# Two-OCO strategy when feasible
def try_two_oco(sym: str, qty: float, t1: float, t2: float, stop: float, tick: float,
                step_qty: float, min_qty: float, min_notional: float) -> tuple[dict | None, dict | None]:
    """
    Attempt two OCOs:
      - q1 -> TP at T1
      - q2 -> TP at T2
    Returns (oco1, oco2) or (None, None) if not feasible.
    """
    if not SPLIT_USE_TWO_OCO:
        return (None, None)

    q1_raw = max(min_qty, qty * max(0.0, min(1.0, SPLIT_Q1_FRACTION)))
    q1 = floor_to_step(q1_raw, step_qty)
    q2 = floor_to_step(qty - q1, step_qty)
    if q1 < min_qty or q2 < min_qty:
        return (None, None)

    # Check notionals
    last = symbol_last_price(sym) or max(t1, t2)
    if q1 * last < min_notional or q2 * last < min_notional:
        return (None, None)

    # Prices rounded
    tp1 = round_to_tick(t1, tick)
    tp2 = round_to_tick(t2, tick)
    stop_trig = round_to_tick(stop, tick)
    stop_lim  = round_to_tick(stop * (1 - STOP_LIMIT_OFFSET), tick)

    oco1 = place_oco_sell(sym, q1, tp1, stop_trig, stop_lim)
    if not oco1:
        return (None, None)
    # If first succeeded, attempt second
    oco2 = place_oco_sell(sym, q2, tp2, stop_trig, stop_lim)
    if not oco2:
        # try to revert to single OCO on full qty? We already consumed q1. Best effort:
        log(f"[TWO-OCO] second leg failed; keeping first OCO only.")
    return (oco1, oco2)

def place_exits_best(sym: str, qty: float, t1: float, t2: float, stop: float, pos: dict) -> None:
    """
    Preferred -> two OCOs (if feasible) else one OCO (T2) else TP limit + virtual stop.
    Updates pos with oco ids or tp order id.
    """
    pos = repair_filters(sym, pos)
    tick = pos.get("tick", 0.0)
    step_qty = pos.get("step_qty", 0.0)
    min_qty = pos.get("min_qty", 0.0)
    min_notional = pos.get("min_notional", MIN_ORDER_USD)

    # Guard: if stop already at/above last price, OCO will reject -> use market exit immediately.
    last = symbol_last_price(sym)
    if last and last <= stop:
        log(f"[EXIT GUARD] {sym} last<=stop -> market exit now.")
        sell_market(sym, qty)
        pos["status"] = "closed"
        pos["closed_at"] = now_str()
        return

    # Try two OCO legs
    oco1, oco2 = try_two_oco(sym, qty, t1, t2, stop, tick, step_qty, min_qty, min_notional)
    if oco1:
        pos["oco_id_t1"] = oco1.get("orderListId")
        if oco2:
            pos["oco_id_t2"] = oco2.get("orderListId")
        else:
            pos["oco_id_t2"] = None
        return

    # Fallback: single OCO at T2
    tp2 = round_to_tick(max(t2, t1), tick)
    stop_trig = round_to_tick(stop, tick)
    stop_lim  = round_to_tick(stop * (1 - STOP_LIMIT_OFFSET), tick)
    oco = place_oco_sell(sym, qty, tp2, stop_trig, stop_lim)
    if oco:
        pos["oco_id"] = oco.get("orderListId")
        return

    # Final fallback: TP limit only; virtual stop handled by maint loop
    tp_only = place_limit_sell(sym, qty, tp2)
    if tp_only:
        pos["tp_order_id"] = tp_only.get("orderId")

# -------- Main --------
def main():
    free, locked, total = get_spot_balance(QUOTE_ASSET)
    if FORCE_EQUITY > 0:
        log(f"Balance: OVERRIDE {FORCE_EQUITY:.2f} {QUOTE_ASSET} | live free={free:.2f}, locked={locked:.2f}, total={total:.2f}")
    else:
        log(f"Balance: live free={free:.2f}, locked={locked:.2f}, total={total:.2f} {QUOTE_ASSET}")

    summary = fetch_summary()
    if not summary:
        return

    state = load_state()
    state["_last_run"] = now_str()
    state["_last_balance"] = {"asset": QUOTE_ASSET, "free": round(free,8), "locked": round(locked,8), "total": round(total,8)}

    # 1) Place candidate stop-limit entries
    if TRADE_CANDS:
        eq_c = equity_for_sizing() * max(0.0, C_RISK_MULT)
        for c in summary.get("candidates", []):
            sym = c["symbol"]
            if not sym.endswith(QUOTE_ASSET):
                continue
            entry = float(c["entry"]); stop = float(c["stop"]); t1=float(c["t1"]); t2=float(c["t2"])
            try:
                f = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}"); continue
            min_qty, step_qty, tick, min_notional = f["min_qty"], f["step_qty"], f["tick"], f["min_notional"]

            qty = compute_qty(entry, stop, eq_c, min_qty, step_qty, min_notional)
            if qty <= 0 and ALLOW_MIN_ORDER:
                qty = floor_to_step(max(min_qty, MIN_ORDER_USD / max(entry,1e-12)), step_qty)

            if qty <= 0 or (qty * entry) < (min_notional - 1e-9):
                log(f"[CAND SKIP] {sym} qty too small")
                continue

            o = place_stop_limit_buy(sym, qty, entry, tick)
            if o:
                state[sym] = {
                    "entry": entry, "stop": stop, "t1": t1, "t2": t2,
                    "filled_qty": 0.0, "t1_filled_qty": 0.0,
                    "status": "pending", "entry_order_id": o.get("orderId"),
                    "type": "C",
                    "tick": tick, "step_qty": step_qty, "min_qty": min_qty, "min_notional": min_notional,
                    "created_at": now_str()
                }

    # 2) Immediate 'B' orders at market
    if summary.get("signals", {}).get("type") == "B":
        eq_b = equity_for_sizing()
        for o in summary.get("orders", []) or []:
            sym = o["symbol"]
            if not sym.endswith(QUOTE_ASSET):
                continue
            entry = float(o["entry"]); stop = float(o["stop"]); t1=float(o["t1"]); t2=float(o["t2"])
            try:
                f = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}"); continue
            min_qty, step_qty, tick, min_notional = f["min_qty"], f["step_qty"], f["tick"], f["min_notional"]

            qty = compute_qty(entry, stop, eq_b, min_qty, step_qty, min_notional)
            if qty <= 0 or (qty * entry) < (min_notional - 1e-9):
                log(f"[BREAKOUT SKIP] {sym} qty too small")
                continue

            m = place_market_buy(sym, qty)
            if m:
                pos = {
                    "entry": entry, "stop": stop, "t1": t1, "t2": t2,
                    "filled_qty": qty, "t1_filled_qty": 0.0,
                    "status": "live", "entry_order_id": m.get("orderId"),
                    "type": "B",
                    "tick": tick, "step_qty": step_qty, "min_qty": min_qty, "min_notional": min_notional,
                    "created_at": now_str()
                }
                place_exits_best(sym, qty, t1, t2, stop, pos)
                state[sym] = pos

    # 3) Maintenance: promote pending, ensure exits, virtual stop if needed
    def maybe_activate_pending(sym: str, pos: dict):
        if pos.get("status") != "pending":
            return
        oid = pos.get("entry_order_id")
        st = order_status(sym, oid)
        if st in ("FILLED","PARTIALLY_FILLED","NEW","PENDING_NEW","EXECUTING","ACCEPTED"):
            try:
                o = client.get_order(symbol=sym, orderId=oid)
                executed_qty = float(o.get("executedQty", "0"))
                orig_qty = float(o.get("origQty", "0.00000000")) or float(pos.get("filled_qty", 0.0))
                fill_pct = executed_qty / orig_qty if orig_qty > 0 else 0.0
            except Exception:
                executed_qty, fill_pct = 0.0, 0.0
            if st == "FILLED" or fill_pct >= PROMOTE_MIN_FILL_PCT:
                pos["filled_qty"] = executed_qty if executed_qty > 0 else pos.get("filled_qty", 0.0)
                pos["status"] = "live"
                place_exits_best(sym, pos["filled_qty"], pos["t1"], pos["t2"], pos["stop"], pos)
                log(f"[PROMOTE] {sym} pending -> live (qty={pos['filled_qty']:.8f})")
        elif st in ("CANCELED","REJECTED","EXPIRED"):
            pos["status"] = "dead"
            log(f"[PENDING DEAD] {sym} status={st}")

    def maybe_virtual_stop(sym: str, pos: dict):
        if pos.get("status") != "live":
            return
        has_exchange_stop = any(k in pos for k in ("oco_id","oco_id_t1","oco_id_t2"))
        if has_exchange_stop:
            return
        last = symbol_last_price(sym)
        stop = float(pos["stop"])
        if last and last <= stop * (1 - PRICE_SLIPPAGE):
            qty = float(pos.get("filled_qty", 0.0))
            if qty > 0:
                sell_market(sym, qty)
                pos["status"] = "closed"
                pos["closed_at"] = now_str()
                log(f"[VSTOP EXIT] {sym} mkt sell qty={qty:.8f} @~{last}")

    for sym, pos in list(state.items()):
        if sym.startswith("_"):
            continue
        pos = repair_filters(sym, pos)
        if pos.get("status") == "pending":
            maybe_activate_pending(sym, pos)
        elif pos.get("status") == "live":
            # If no exits yet (e.g., OCO failed earlier), try again
            if not any(k in pos for k in ("oco_id","oco_id_t1","oco_id_t2","tp_order_id")):
                place_exits_best(sym, pos["filled_qty"], pos["t1"], pos["t2"], pos["stop"], pos)
            maybe_virtual_stop(sym, pos)
        state[sym] = pos

    save_state(state)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("UNCAUGHT ERROR:\n" + traceback.format_exc())