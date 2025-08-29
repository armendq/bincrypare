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

client = Client(API_KEY, API_SECRET)

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[exec {ts}] {msg}", flush=True)

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

def get_equity_usd() -> float:
    if FORCE_EQUITY > 0:
        return FORCE_EQUITY
    acct = client.get_account()
    balmap = {b["asset"]: float(b["free"]) + float(b["locked"]) for b in acct["balances"]}
    return balmap.get(QUOTE_ASSET, 0.0)

def place_order(sym, entry, stop, t1, t2):
    eq = get_equity_usd()
    log(f"Equity available ({QUOTE_ASSET}): {eq:.2f}")

    risk = eq * 0.012
    risk_per_unit = max(entry - stop, entry * 0.002)
    raw_qty = risk / risk_per_unit
    raw_qty = math.floor(raw_qty * 100) / 100
    notional = raw_qty * entry
    if notional < 5.0:
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

    # always print balance, even if no signals
    eq = get_equity_usd()
    log(f"Balance check: {eq:.2f} {QUOTE_ASSET}")

    if S.get("signals", {}).get("type") != "B":
        log("Hold and wait.")
        return

    for o in S.get("orders", []):
        sym = o["symbol"]
        if not sym.endswith(QUOTE_ASSET):
            continue
        place_order(sym, o["entry"], o["stop"], o["t1"], o["t2"])

if __name__ == "__main__":
    main()
