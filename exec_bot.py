#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, math, requests
from pathlib import Path
from datetime import datetime, timezone
from binance.client import Client

SUMMARY_URL = "https://raw.githubusercontent.com/armendq/bincrypare/main/public_runs/latest/summary.json"
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDC")
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"
FORCE_EQUITY = float(os.getenv("FORCE_EQUITY_USD", "0"))

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

STATE_DIR = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "open_positions.json"

RISK_PCT = 0.012
MIN_NOTIONAL = 5.0
DUP_ENTRY_TOL = 0.002

client = Client(API_KEY, API_SECRET)

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

def get_spot_usdc_balance() -> tuple[float, float, float]:
    """Return (free, locked, total) for QUOTE_ASSET on Spot."""
    acct = client.get_account()
    free = locked = 0.0
    for b in acct.get("balances", []):
        if b.get("asset") == QUOTE_ASSET:
            free = float(b.get("free", "0"))
            locked = float(b.get("locked", "0"))
            break
    return free, locked, free + locked

def equity_for_sizing() -> float:
    """Use override if set; else use FREE balance (tradable)."""
    if FORCE_EQUITY > 0:
        return FORCE_EQUITY
    free, _, _ = get_spot_usdc_balance()
    return free

def place_order(sym: str, entry: float, stop: float, t1: float, t2: float) -> float:
    # print balances every time
    free, locked, total = get_spot_usdc_balance()
    if FORCE_EQUITY > 0:
        log(f"Balance check: OVERRIDE {FORCE_EQUITY:.2f} {QUOTE_ASSET} | live free={free:.2f}, locked={locked:.2f}, total={total:.2f}")
    else:
        log(f"Balance check: live free={free:.2f}, locked={locked:.2f}, total={total:.2f} {QUOTE_ASSET}")

    eq = equity_for_sizing()
    log(f"Equity used for sizing ({QUOTE_ASSET}): {eq:.2f}")

    risk = eq * RISK_PCT
    risk_per_unit = max(entry - stop, entry * 0.002)
    if risk_per_unit <= 0:
        log(f"[SKIP] {sym} invalid risk_per_unit")
        return 0.0

    raw_qty = risk / risk_per_unit
    raw_qty = max(raw_qty, MIN_NOTIONAL / max(entry, 1e-12))
    # round down a bit to avoid precision issues
    raw_qty = math.floor(raw_qty * 1000000) / 1000000

    notional = raw_qty * entry
    if notional < MIN_NOTIONAL:
        raw_qty = MIN_NOTIONAL / entry
        raw_qty = math.floor(raw_qty * 1000000) / 1000000
        notional = raw_qty * entry

    if DRY_RUN:
        log(f"[DRY ENTRY] {sym} qty={raw_qty:.6f} notional≈{notional:.2f} entry={entry:.8f} stop={stop:.8f} t1={t1:.8f} t2={t2:.8f}")
        return raw_qty

    try:
        order = client.order_market_buy(symbol=sym, quantity=raw_qty)
        log(f"[LIVE ENTRY] {sym} market buy qty={raw_qty:.6f} notional≈{notional:.2f}")
    except Exception as e:
        log(f"[ENTRY ERROR] {sym}: {e}")
        return 0.0

    return raw_qty

def main():
    S = fetch_summary()
    if not S:
        return

    state = load_state()
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

    for o in orders:
        sym = o["symbol"]
        if not sym.endswith(QUOTE_ASSET):
            continue

        entry = float(o["entry"]); stop = float(o["stop"])
        t1 = float(o["t1"]); t2 = float(o["t2"])

        prev = state.get(sym)
        if prev and prev.get("status") != "closed":
            prev_entry = float(prev.get("entry", entry))
            if near(prev_entry, entry, DUP_ENTRY_TOL):
                log(f"[SKIP] {sym} duplicate near entry.")
                continue

        qty = place_order(sym, entry, stop, t1, t2)
        if qty <= 0:
            continue

        state[sym] = {
            "entry": entry, "stop": stop, "t1": t1, "t2": t2,
            "filled_qty": qty, "t1_filled_qty": 0.0,
            "last_trail_update_ts": None, "status": "live",
        }

    save_state(state)

if __name__ == "__main__":
    main()
