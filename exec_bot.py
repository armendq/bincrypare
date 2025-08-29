#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Execution layer – consumes summary.json and places trades on Binance Spot

- Quote asset = USDC (default, override with QUOTE_ASSET env).
- FORCE_EQUITY_USD env lets you simulate small accounts (e.g. 20).
- Min notional $5 enforced.
- DRY_RUN = "1" → only logs, DRY_RUN = "0" → real orders.
"""

import os, json, time, math, requests
from datetime import datetime, timezone
from binance.client import Client

# ---------------------- CONFIG ----------------------
TZ_NAME = "Europe/Prague"
SUMMARY_URL = "https://raw.githubusercontent.com/armendq/bincrypare/main/public_runs/latest/summary.json"

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

DRY_RUN = os.getenv("DRY_RUN", "1") == "1"
FORCE_EQUITY = float(os.getenv("FORCE_EQUITY_USD", "0"))
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDC")

client = Client(API_KEY, API_SECRET)

# ---------------------- HELPERS ----------------------
def log(msg):
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[exec {ts}] {msg}", flush=True)

def fetch_summary():
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

def get_equity_usd():
    if FORCE_EQUITY > 0:
        return FORCE_EQUITY
    acct = client.get_account()
    balmap = {b["asset"]: float(b["free"]) + float(b["locked"]) for b in acct["balances"]}
    return balmap.get(QUOTE_ASSET, 0.0)

# ---------------------- CORE ----------------------
def place_order(sym, entry, stop, t1, t2):
    log(f"Signal {sym}: entry={entry}, stop={stop}, t1={t1}, t2={t2}")
    eq = get_equity_usd()
    risk = eq * 0.012
    risk_per_unit = max(entry - stop, entry * 0.002)
    raw_qty = risk / risk_per_unit
    # round to 2 decimals for safety
    raw_qty = math.floor(raw_qty * 100) / 100

    notional = raw_qty * entry
    if notional < 5.0:  # Binance min notional
        notional = 5.0
        raw_qty = notional / entry

    if DRY_RUN:
        log(f"[DRY ENTRY] {sym} qty={raw_qty:.4f} entry={entry}")
        return

    try:
        order = client.create_order(
            symbol=sym,
            side="BUY",
            type="MARKET",
            quantity=raw_qty
        )
        log(f"Order placed: {order}")
    except Exception as e:
        log(f"ORDER ERROR {sym}: {e}")

def main():
    S = fetch_summary()
    if not S:
        return
    if S["signals"]["type"] != "B":
        log("Hold and wait.")
        return

    for o in S["orders"]:
        sym = o["symbol"]
        if not sym.endswith(QUOTE_ASSET):
            continue
        place_order(sym, o["entry"], o["stop"], o["t1"], o["t2"])

if __name__ == "__main__":
    main()
