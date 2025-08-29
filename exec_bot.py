#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, math, requests
from pathlib import Path
from datetime import datetime, timezone
from binance.client import Client

# -------- Config
SUMMARY_URL = "https://raw.githubusercontent.com/armendq/bincrypare/main/public_runs/latest/summary.json"
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDC")
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"
FORCE_EQUITY = float(os.getenv("FORCE_EQUITY_USD", "0"))

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

STATE_DIR = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "open_positions.json"

RISK_PCT = 0.012      # 1.2% risk
MIN_NOTIONAL = 5.0    # Binance min order notional (approx)
DUP_ENTRY_TOL = 0.002 # ±0.2% duplicate guard

# -------- Utils
def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[exec {ts}] {msg}", flush=True)

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_state(state: dict) -> None:
    tmp = STATE_DIR / "open_positions.tmp"
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)

def fetch_summary() -> dict | None:
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

def near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(1e-12, b)

# -------- Binance helpers
client = Client(API_KEY, API_SECRET)

def get_equity_usd() -> float:
    if FORCE_EQUITY > 0:
        return FORCE_EQUITY
    acct = client.get_account()
    balmap = {b["asset"]: float(b["free"]) + float(b["locked"]) for b in acct["balances"]}
    return balmap.get(QUOTE_ASSET, 0.0)

def get_symbol_filters(symbol: str) -> tuple[float, float, float]:
    info = client.get_symbol_info(symbol)
    lot = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
    pricef = next(f for f in info["filters"] if f["filterType"] == "PRICE_FILTER")
    min_qty = float(lot["minQty"])
    step_qty = float(lot["stepSize"])
    tick = float(pricef["tickSize"])
    return min_qty, step_qty, tick

def floor_to_step(value: float, step: float) -> float:
    if step <= 0: return value
    return math.floor(value / step) * step

# -------- Sizing
def compute_qty(entry: float, stop: float, equity: float, min_qty: float, step_qty: float) -> float:
    risk_dollars = equity * RISK_PCT
    risk_per_unit = max(entry - stop, entry * 0.002)
    if risk_per_unit <= 0: return 0.0
    raw_qty = risk_dollars / risk_per_unit
    # enforce min notional $5
    if raw_qty * entry < MIN_NOTIONAL:
        raw_qty = MIN_NOTIONAL / entry
    qty = floor_to_step(raw_qty, step_qty)
    if qty < min_qty: return 0.0
    return qty

# -------- Core
def main():
    S = fetch_summary()
    if not S:
        return

    state = load_state()
    # optional heartbeat so you can see file updates; comment out if you don’t want per-run changes
    state["_last_run"] = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

    if S.get("signals", {}).get("type") != "B":
        log("Hold and wait.")
        save_state(state)
        return

    orders = S.get("orders", []) or []
    if not orders:
        log("No orders in B signal.")
        save_state(state)
        return

    equity = get_equity_usd()
    log(f"Equity basis ({QUOTE_ASSET}): {equity:.2f}")

    for o in orders:
        symbol = o["symbol"]
        if not symbol.endswith(QUOTE_ASSET):
            continue

        entry = float(o["entry"])
        stop = float(o["stop"])
        t1 = float(o["t1"])
        t2 = float(o["t2"])

        # duplicate protection
        prev = state.get(symbol)
        if prev and prev.get("status") != "closed":
            prev_entry = float(prev.get("entry", entry))
            if near(prev_entry, entry, DUP_ENTRY_TOL):
                log(f"[SKIP] {symbol} duplicate near entry.")
                continue

        try:
            min_qty, step_qty, tick = get_symbol_filters(symbol)
        except Exception as e:
            log(f"[FILTER] {symbol} error: {e}")
            continue

        qty = compute_qty(entry, stop, equity, min_qty, step_qty)
        if qty <= 0:
            log(f"[SKIP] {symbol} qty too small.")
            continue

        if DRY_RUN:
            log(f"[DRY ENTRY] {symbol} qty={qty:.8f} entry={entry:.8f} stop={stop:.8f} t1={t1:.8f} t2={t2:.8f}")
        else:
            try:
                client.order_market_buy(symbol=symbol, quantity=qty)
                log(f"[LIVE ENTRY] {symbol} market buy qty={qty:.8f}")
            except Exception as e:
                log(f"[ENTRY ERROR] {symbol}: {e}")
                continue

        # record/update state
        state[symbol] = {
            "entry": entry,
            "stop": stop,
            "t1": t1,
            "t2": t2,
            "filled_qty": qty,
            "t1_filled_qty": 0.0,
            "last_trail_update_ts": None,
            "status": "live",
        }

    save_state(state)

if __name__ == "__main__":
    main()
