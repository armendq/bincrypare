#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exec_bot.py — USDC spot executor (live)

What this does
- Trades USDC symbols from summary.json.
- Trades "C" candidates with stop-limit BUYs (reduced risk).
- Trades "B" with market BUYs.
- Exits prefer two OCO legs (T1/T2). Fallbacks: one OCO, or TP limit + virtual stop.
- Enforces: max open positions, min-notional, precision, breakeven after +ATR, time-based promotion/cancel/cut.
- Persists state/open_positions.json.

Environment (read from process env)
- SUMMARY_URL                      (required; file:// or https://)
- BINANCE_API_KEY, BINANCE_API_SECRET
- QUOTE_ASSET                      default USDC
- DRY_RUN                          1/0 (set 0 for live)
- FORCE_EQUITY_USD                 numeric; if >0, use this for sizing (start with 400)
- RISK_PCT                         default 0.012 (1.2%)
- TRADE_CANDIDATES                 1/0
- C_RISK_MULT                      default 0.6 (risk fraction vs RISK_PCT for candidates)
- STOP_LIMIT_OFFSET                default 0.001 (0.1% above entry)
- PRICE_SLIPPAGE                   default 0.003 (virtual stop cushion)
- ALLOW_MIN_ORDER                  1/0
- MIN_ORDER_USD                    default 6

Risk/ops controls
- MAX_OPEN_POSITIONS               default 2 (live+pending count)
- PROMOTE_PENDING_AFTER_MIN        default 8
- CANCEL_STALE_PENDING_MIN         default 45
- CUT_LOSS_MIN                     default 60
- CUT_LOSS_PCT                     default 0.01  (cut if age>min and price<-1% vs entry)
- BE_PROFIT_ATR                    default 0.8   (breakeven after +0.8 ATR)
"""

import os, json, time, math, requests, traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta

from binance.client import Client
from binance.enums import (
    SIDE_BUY, SIDE_SELL, TIME_IN_FORCE_GTC,
    ORDER_TYPE_MARKET, ORDER_TYPE_STOP_LOSS_LIMIT, ORDER_TYPE_LIMIT
)

# -------- ENV --------
SUMMARY_URL           = os.getenv("SUMMARY_URL", "https://raw.githubusercontent.com/armendq/bincrypare/main/public_runs/latest/summary.json")
QUOTE_ASSET           = os.getenv("QUOTE_ASSET", "USDC")
DRY_RUN               = os.getenv("DRY_RUN", "1") == "1"
FORCE_EQUITY          = float(os.getenv("FORCE_EQUITY_USD", "0"))  # set to 400 to start
RISK_PCT              = float(os.getenv("RISK_PCT", "0.012"))

TRADE_CANDS           = os.getenv("TRADE_CANDIDATES", "1") == "1"
C_RISK_MULT           = float(os.getenv("C_RISK_MULT", "0.6"))     # sizing multiplier for C

STOP_LIMIT_OFFSET     = float(os.getenv("STOP_LIMIT_OFFSET", "0.001"))
PRICE_SLIPPAGE        = float(os.getenv("PRICE_SLIPPAGE", "0.003"))

ALLOW_MIN_ORDER       = os.getenv("ALLOW_MIN_ORDER", "1") == "1"
MIN_ORDER_USD         = float(os.getenv("MIN_ORDER_USD", "6.0"))

MAX_OPEN_POSITIONS    = int(os.getenv("MAX_OPEN_POSITIONS", "2"))
PROMOTE_MIN           = int(os.getenv("PROMOTE_PENDING_AFTER_MIN", "8"))
CANCEL_STALE_MIN      = int(os.getenv("CANCEL_STALE_PENDING_MIN", "45"))
CUT_LOSS_MIN          = int(os.getenv("CUT_LOSS_MIN", "60"))
CUT_LOSS_PCT          = float(os.getenv("CUT_LOSS_PCT", "0.01"))
BE_PROFIT_ATR         = float(os.getenv("BE_PROFIT_ATR", "0.8"))

STATE_DIR  = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "open_positions.json"

API_KEY   = os.getenv("BINANCE_API_KEY", "")
API_SECRET= os.getenv("BINANCE_API_SECRET", "")

client = Client(API_KEY, API_SECRET)

# -------- Utils --------
def now() -> datetime:
    return datetime.now(timezone.utc).astimezone()

def now_str() -> str:
    return now().strftime("%Y-%m-%d %H:%M:%S")

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

def parse_iso(dt_str: str) -> datetime:
    try:
        return datetime.fromisoformat(dt_str).astimezone()
    except Exception:
        # Fallback parse of "%Y-%m-%d %H:%M:%S"
        try:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).astimezone()
        except Exception:
            return now()

def age_minutes(dt_str: str) -> float:
    try:
        return (now() - parse_iso(dt_str)).total_seconds() / 60.0
    except Exception:
        return 0.0

# -------- I/O --------
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

# -------- Filters & rounding --------
def floor_to_step(v: float, step: float) -> float:
    if step <= 0:
        return v
    return math.floor(v / step) * step

def round_to_tick(p: float, tick: float) -> float:
    if tick <= 0:
        return p
    return floor_to_step(p, tick)

def get_symbol_filters(symbol: str) -> dict:
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
    rpu = max(entry - stop, entry * 0.002)  # avoid absurd stops
    if rpu <= 0:
        return 0.0
    raw_qty = max(risk_dollars / rpu, min_notional / max(entry, 1e-12))
    qty = floor_to_step(raw_qty, step_qty)
    if qty < min_qty:
        return 0.0
    return qty

# -------- Market data / orders --------
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

def cancel_oco(symbol: str, order_list_id) -> None:
    if DRY_RUN:
        log(f"[CANCEL OCO dry] {symbol} listId={order_list_id}")
        return
    try:
        client.cancel_oco_order(symbol=symbol, orderListId=order_list_id)
        log(f"[CANCEL OCO] {symbol} listId={order_list_id}")
    except Exception as e:
        log(f"[CANCEL OCO ERROR] {symbol}: {e}")

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

# -------- Exit placement orchestration --------
def try_two_oco(sym: str, qty: float, t1: float, t2: float, stop: float, tick: float,
                step_qty: float, min_qty: float, min_notional: float) -> tuple[dict | None, dict | None]:
    # Split
    q1_raw = max(min_qty, qty * 0.5)
    q1 = floor_to_step(q1_raw, step_qty)
    q2 = floor_to_step(qty - q1, step_qty)
    if q1 < min_qty or q2 < min_qty:
        return (None, None)

    last = symbol_last_price(sym) or max(t1, t2)
    if q1 * last < min_notional or q2 * last < min_notional:
        return (None, None)

    tp1 = round_to_tick(t1, tick)
    tp2 = round_to_tick(t2, tick)
    stop_trig = round_to_tick(stop, tick)
    stop_lim  = round_to_tick(stop * (1 - STOP_LIMIT_OFFSET), tick)

    oco1 = place_oco_sell(sym, q1, tp1, stop_trig, stop_lim)
    if not oco1:
        return (None, None)
    oco2 = place_oco_sell(sym, q2, tp2, stop_trig, stop_lim)
    if not oco2:
        log(f"[TWO-OCO] second leg failed; keeping first OCO only.")
    return (oco1, oco2)

def place_exits_best(sym: str, qty: float, t1: float, t2: float, stop: float, pos: dict) -> None:
    pos = ensure_filters(sym, pos)
    tick = pos["tick"]; step_qty = pos["step_qty"]; min_qty = pos["min_qty"]; min_notional = pos["min_notional"]

    last = symbol_last_price(sym)
    if last and last <= stop:
        # Already under stop -> exit now to avoid OCO rejection
        log(f"[EXIT GUARD] {sym} last<=stop -> market exit.")
        sell_market(sym, qty)
        pos["status"] = "closed"; pos["closed_at"] = now_str()
        return

    o1, o2 = try_two_oco(sym, qty, t1, t2, stop, tick, step_qty, min_qty, min_notional)
    if o1:
        pos["oco_id_t1"] = o1.get("orderListId")
        if o2:
            pos["oco_id_t2"] = o2.get("orderListId")
        return

    tp2 = round_to_tick(max(t2, t1), tick)
    stop_trig = round_to_tick(stop, tick)
    stop_lim  = round_to_tick(stop * (1 - STOP_LIMIT_OFFSET), tick)
    oco = place_oco_sell(sym, qty, tp2, stop_trig, stop_lim)
    if oco:
        pos["oco_id"] = oco.get("orderListId")
        return

    tp_only = place_limit_sell(sym, qty, tp2)
    if tp_only:
        pos["tp_order_id"] = tp_only.get("orderId")
        # Virtual stop active when only TP is on-book
        pos["virtual_stop"] = stop

# -------- Guards --------
def ensure_filters(sym: str, pos: dict) -> dict:
    need = any(k not in pos for k in ("tick","step_qty","min_qty","min_notional"))
    if not need:
        return pos
    try:
        f = get_symbol_filters(sym)
        pos["tick"], pos["step_qty"], pos["min_qty"], pos["min_notional"] = f["tick"], f["step_qty"], f["min_qty"], f["min_notional"]
    except Exception:
        # best effort; defaults
        pos.setdefault("tick", 0.00000001)
        pos.setdefault("step_qty", 0.00000001)
        pos.setdefault("min_qty", 0.0)
        pos.setdefault("min_notional", MIN_ORDER_USD)
    return pos

def open_positions_count(state: dict) -> int:
    n = 0
    for k, v in state.items():
        if k.startswith("_"):
            continue
        if v.get("status") in ("pending","live"):
            n += 1
    return n

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

    # Respect max open positions
    def can_open_more() -> bool:
        cnt = open_positions_count(state)
        if cnt >= MAX_OPEN_POSITIONS:
            log(f"[CAP] Open positions {cnt} >= max {MAX_OPEN_POSITIONS}. Skipping new entries.")
            return False
        return True

    eq_total = equity_for_sizing()

    # 1) Place candidate stop-limit entries
    if TRADE_CANDS:
        eq_c = eq_total * max(0.0, C_RISK_MULT)
        for c in summary.get("candidates", []):
            if not can_open_more():
                break
            sym = c.get("symbol", "")
            if not sym.endswith(QUOTE_ASSET):
                continue
            entry = float(c["entry"]); stop = float(c["stop"]); t1=float(c["t1"]); t2=float(c["t2"])
            atr = float(c.get("atr", 0.0))
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
                    "entry": entry, "stop": stop, "t1": t1, "t2": t2, "atr": atr,
                    "filled_qty": 0.0, "t1_filled_qty": 0.0,
                    "status": "pending", "entry_order_id": o.get("orderId"),
                    "type": "C",
                    "tick": tick, "step_qty": step_qty, "min_qty": min_qty, "min_notional": min_notional,
                    "created_at": now_str()
                }

    # 2) Immediate 'B' orders at market
    if summary.get("signals", {}).get("type") == "B":
        for o in summary.get("orders", []) or []:
            if not can_open_more():
                break
            sym = o.get("symbol", "")
            if not sym.endswith(QUOTE_ASSET):
                continue
            entry = float(o["entry"]); stop = float(o["stop"]); t1=float(o["t1"]); t2=float(o["t2"])
            atr  = float(o.get("atr", 0.0))
            try:
                f = get_symbol_filters(sym)
            except Exception as e:
                log(f"[FILTER] {sym} {e}"); continue
            min_qty, step_qty, tick, min_notional = f["min_qty"], f["step_qty"], f["tick"], f["min_notional"]

            qty = compute_qty(entry, stop, eq_total, min_qty, step_qty, min_notional)
            if qty <= 0 or (qty * entry) < (min_notional - 1e-9):
                log(f"[BREAKOUT SKIP] {sym} qty too small")
                continue

            m = place_market_buy(sym, qty)
            if m:
                pos = {
                    "entry": entry, "stop": stop, "t1": t1, "t2": t2, "atr": atr,
                    "filled_qty": qty, "t1_filled_qty": 0.0,
                    "status": "live", "entry_order_id": m.get("orderId"),
                    "type": "B",
                    "tick": tick, "step_qty": step_qty, "min_qty": min_qty, "min_notional": min_notional,
                    "created_at": now_str()
                }
                place_exits_best(sym, qty, t1, t2, stop, pos)
                state[sym] = pos

    # 3) Maintenance loop
    def maybe_activate_pending(sym: str, pos: dict):
        if pos.get("status") != "pending":
            return
        # Promote to market if price has moved through entry and order sits longer than PROMOTE_MIN
        try:
            last = symbol_last_price(sym)
        except Exception:
            last = 0.0
        if last and last >= pos["entry"] * (1 + STOP_LIMIT_OFFSET * 0.5) and age_minutes(pos["created_at"]) >= PROMOTE_MIN:
            # Market buy remaining (best-effort)
            try:
                o = client.get_order(symbol=sym, orderId=pos["entry_order_id"])
                executed = float(o.get("executedQty", "0"))
                orig     = float(o.get("origQty", "0"))
            except Exception:
                executed, orig = 0.0, 0.0
            remain = max(0.0, orig - executed) if orig > 0 else 0.0
            if remain > 0:
                mb = place_market_buy(sym, remain)
                if mb:
                    pos["filled_qty"] = executed + remain
            pos["status"] = "live"
            place_exits_best(sym, pos.get("filled_qty", 0.0), pos["t1"], pos["t2"], pos["stop"], pos)
            log(f"[PROMOTE] {sym} pending -> live (market promotion)")
            return

        # Cancel stale pending
        if age_minutes(pos["created_at"]) >= CANCEL_STALE_MIN:
            try:
                client.cancel_order(symbol=sym, orderId=pos["entry_order_id"])
                log(f"[PENDING CANCEL] {sym} stale > {CANCEL_STALE_MIN}m")
            except Exception as e:
                log(f"[PENDING CANCEL ERROR] {sym}: {e}")
            pos["status"] = "dead"

        # Normal fill path
        st = order_status(sym, pos.get("entry_order_id"))
        if st in ("FILLED","PARTIALLY_FILLED","NEW","PENDING_NEW","EXECUTING","ACCEPTED"):
            try:
                o = client.get_order(symbol=sym, orderId=pos["entry_order_id"])
                executed_qty = float(o.get("executedQty", "0"))
                orig_qty = float(o.get("origQty", "0.00000000")) or float(pos.get("filled_qty", 0.0))
                fill_pct = executed_qty / orig_qty if orig_qty > 0 else 0.0
            except Exception:
                executed_qty, fill_pct = 0.0, 0.0
            if st == "FILLED" or fill_pct >= 0.60:
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
        if has_exchange_stop and "virtual_stop" not in pos:
            return
        last = symbol_last_price(sym)
        trigger = pos.get("virtual_stop", pos["stop"])
        if last and last <= trigger * (1 - PRICE_SLIPPAGE):
            qty = float(pos.get("filled_qty", 0.0))
            if qty > 0:
                sell_market(sym, qty)
                pos["status"] = "closed"; pos["closed_at"] = now_str()
                log(f"[VSTOP EXIT] {sym} mkt sell qty={qty:.8f} @~{last}")

    def maybe_breakeven(sym: str, pos: dict):
        if pos.get("status") != "live":
            return
        if pos.get("be_done"):
            return
        atr = float(pos.get("atr", 0.0))
        if atr <= 0:
            return
        last = symbol_last_price(sym)
        if last and last >= pos["entry"] + BE_PROFIT_ATR * atr:
            # Only safe to adjust if using virtual stop. If on-exchange OCO, skip to avoid cancel/recreate churn.
            if not any(k in pos for k in ("oco_id","oco_id_t1","oco_id_t2")):
                pos["virtual_stop"] = max(pos.get("virtual_stop", pos["stop"]), pos["entry"])
                pos["be_done"] = True
                log(f"[BREAKEVEN] {sym} virtual_stop -> {pos['virtual_stop']:.8f}")

    def maybe_time_cut(sym: str, pos: dict):
        if pos.get("status") != "live":
            return
        if age_minutes(pos.get("created_at", now_str())) < CUT_LOSS_MIN:
            return
        last = symbol_last_price(sym)
        if not last:
            return
        if last <= pos["entry"] * (1 - CUT_LOSS_PCT):
            qty = float(pos.get("filled_qty", 0.0))
            if qty > 0:
                # Try to cancel any OCO first to avoid conflicts
                for oid_key in ("oco_id","oco_id_t1","oco_id_t2"):
                    if oid_key in pos and pos[oid_key]:
                        cancel_oco(sym, pos[oid_key])
                sell_market(sym, qty)
                pos["status"] = "closed"; pos["closed_at"] = now_str()
                log(f"[TIME CUT] {sym} age>{CUT_LOSS_MIN}m, cut loss at ~{last}")

    # Iterate positions
    for sym, pos in list(state.items()):
        if sym.startswith("_"):
            continue
        pos = ensure_filters(sym, pos)
        if pos.get("status") == "pending":
            maybe_activate_pending(sym, pos)
        elif pos.get("status") == "live":
            # Ensure exits if none yet
            if not any(k in pos for k in ("oco_id","oco_id_t1","oco_id_t2","tp_order_id","virtual_stop")):
                place_exits_best(sym, pos["filled_qty"], pos["t1"], pos["t2"], pos["stop"], pos)
            maybe_breakeven(sym, pos)
            maybe_virtual_stop(sym, pos)
            maybe_time_cut(sym, pos)
        state[sym] = pos

    save_state(state)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("UNCAUGHT ERROR:\n" + traceback.format_exc())