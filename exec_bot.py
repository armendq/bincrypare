#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

TZ = ZoneInfo("Europe/Prague")
SUMMARY_URL = "https://raw.githubusercontent.com/armendq/revolut_crypto_mapping/main/public_runs/latest/summary.json"

STATE_DIR = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "open_positions.json"

RISK_PCT = 0.012
EXPO_CAP_OK = 0.60
EXPO_CAP_BAD = 0.30
DUP_ENTRY_TOL = 0.002
SLIPPAGE_ATR_MULT = 0.8
TRAIL_ATR_MULT = 1.2
TRAIL_LOOKBACK_BARS = 3

def log(msg):
    ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{ts}] {msg}")

def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_state(state):
    tmp = STATE_DIR / "open_positions.tmp"
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)

def fetch_summary():
    for attempt in (1, 2):
        try:
            r = requests.get(SUMMARY_URL, timeout=15)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        if attempt == 1:
            time.sleep(60)
    print("Hold and wait. Fetch failed twice.")
    return None

def floor_to_step(value, step):
    if step <= 0:
        return value
    return math.floor(value / step) * step

def round_to_step(value, step):
    if step <= 0:
        return value
    return floor_to_step(value + 1e-15, step)

def get_symbol_filters(client, symbol):
    info = client.get_symbol_info(symbol)
    if not info:
        raise RuntimeError(f"Missing symbol info for {symbol}")
    lot = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
    pricef = next(f for f in info["filters"] if f["filterType"] == "PRICE_FILTER")
    min_qty = float(lot["minQty"])
    step_qty = float(lot["stepSize"])
    tick = float(pricef["tickSize"])
    return min_qty, step_qty, tick

def get_last_price(client, symbol):
    t = client.get_symbol_ticker(symbol=symbol)
    return float(t["price"])

def account_equity_usdt(client):
    acct = client.get_account()
    balances = {b["asset"]: (float(b["free"]), float(b["locked"])) for b in acct["balances"]}
    tickers = {t["symbol"]: float(t["price"]) for t in client.get_all_tickers()}
    def px(asset):
        if asset == "USDT": return 1.0
        return tickers.get(f"{asset}USDT", 0.0)
    equity = 0.0
    for asset, (free, locked) in balances.items():
        total = free + locked
        if total > 0:
            equity += total * px(asset)
    cash = balances.get("USDT", (0.0, 0.0))[0]
    return equity, cash

def current_gross_exposure(state):
    total = 0.0
    for pos in state.values():
        if pos.get("status") == "closed": continue
        total += float(pos["entry"]) * float(pos.get("filled_qty", 0.0))
    return total

def size_order(entry, stop, equity_usd, cash_usdt, min_qty, step_qty):
    risk_dollars = equity_usd * RISK_PCT
    risk_per_unit = max(entry - stop, entry * 0.002)
    if risk_per_unit <= 0: return 0.0
    raw_qty = risk_dollars / risk_per_unit
    qty = floor_to_step(raw_qty, step_qty)
    if qty * entry > cash_usdt * 0.98 and entry > 0:
        qty = floor_to_step((cash_usdt * 0.98) / entry, step_qty)
    if qty < min_qty: return 0.0
    return qty

def exposure_cap(equity_usd, regime_ok):
    return equity_usd * (EXPO_CAP_OK if regime_ok else EXPO_CAP_BAD)

def near(a, b, tol): return abs(a - b) <= (tol * max(1e-12, b))

def get_klines_1h(client, symbol, limit=50):
    return client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=limit)

def last3_low_close_ts(klines):
    if len(klines) < 3: return None, None, None
    lows = [float(k[3]) for k in klines[-TRAIL_LOOKBACK_BARS:]]
    last = klines[-1]
    return min(lows), float(last[4]), int(last[6])

def assume_t1_if_hit(client, state, symbol, dry_run):
    pos = state[symbol]
    if pos.get("t1_filled_qty", 0.0) > 0: return
    last = get_last_price(client, symbol)
    if last >= float(pos["t1"]):
        pos["t1_filled_qty"] = float(pos["filled_qty"]) / 2.0
        if dry_run:
            log(f"[ASSUME T1] {symbol} price >= T1. Marked half as sold. Move stop to break-even next.")
        else:
            pass  # OCO should have executed; we will set a BE stop for the remainder in trailing step

def manage_trailing(client, state, symbol, dry_run):
    pos = state[symbol]
    if pos.get("status") == "closed": return
    entry = float(pos["entry"])
    stop = float(pos["stop"])
    atr = float(pos.get("atr", 0.0))
    filled = float(pos.get("filled_qty", 0.0))
    t1_filled = float(pos.get("t1_filled_qty", 0.0))
    qty_left = max(0.0, filled - t1_filled)
    if qty_left <= 0: return

    # after T1 or +1R: ensure BE stop
    r1 = entry + (entry - stop)
    last_price = get_last_price(client, symbol)
    be_needed = (last_price >= r1) or (t1_filled > 0 and stop < entry)
    if be_needed and stop < entry:
        new_stop = entry
        min_qty, step_qty, tick = get_symbol_filters(client, symbol)
        new_stop = round_to_step(new_stop, tick)
        if dry_run:
            log(f"[BE DRY] {symbol} move stop {stop:.8f} -> {new_stop:.8f} for qty {qty_left:.8f}")
        else:
            try:
                client.create_order(
                    symbol=symbol, side="SELL", type="STOP_LOSS_LIMIT",
                    quantity=floor_to_step(qty_left, step_qty),
                    price=f"{new_stop:.8f}",
                    stopPrice=f"{new_stop:.8f}",
                    timeInForce="GTC"
                )
                log(f"[BE LIVE] {symbol} stop set to BE {new_stop:.8f} qty {qty_left:.8f}")
            except Exception as e:
                log(f"[BE ERROR] {symbol} {e}")
        pos["stop"] = new_stop

    # trail once per 1h bar
    try:
        kl = get_klines_1h(client, symbol, limit=50)
    except Exception as e:
        log(f"[KLINES ERROR] {symbol} {e}")
        return
    low3, last_close, close_ts = last3_low_close_ts(kl)
    if low3 is None: return
    last_trail_ts = pos.get("last_trail_update_ts")
    if last_trail_ts and int(last_trail_ts) >= int(close_ts): return

    cand1 = low3
    cand2 = last_close - TRAIL_ATR_MULT * atr
    new_stop = max(stop, cand1, cand2)
    min_qty, step_qty, tick = get_symbol_filters(client, symbol)
    new_stop = round_to_step(new_stop, tick)

    if new_stop <= stop + 1e-12: return

    if dry_run:
        log(f"[TRAIL DRY] {symbol} stop {stop:.8f} -> {new_stop:.8f}")
    else:
        try:
            client.create_order(
                symbol=symbol, side="SELL", type="STOP_LOSS_LIMIT",
                quantity=floor_to_step(qty_left, step_qty),
                price=f"{new_stop:.8f}",
                stopPrice=f"{new_stop:.8f}",
                timeInForce="GTC"
            )
            log(f"[TRAIL LIVE] {symbol} stop -> {new_stop:.8f} qty {qty_left:.8f}")
        except Exception as e:
            log(f"[TRAIL ERROR] {symbol} {e}")
            return
    pos["stop"] = new_stop
    pos["last_trail_update_ts"] = int(close_ts)

def place_entry_and_exits(client, order, qty, dry_run):
    symbol = order["symbol"]
    entry = float(order["entry"])
    stop  = float(order["stop"])
    t1    = float(order["t1"])
    t2    = float(order["t2"])
    min_qty, step_qty, tick = get_symbol_filters(client, symbol)
    entry = round_to_step(entry, tick)
    stop  = round_to_step(stop,  tick)
    t1    = round_to_step(t1,    tick)
    t2    = round_to_step(t2,    tick)

    qty1 = floor_to_step(qty / 2.0, step_qty)
    qty2 = floor_to_step(qty - qty1, step_qty)

    last = get_last_price(client, symbol)
    use_market = last >= entry

    if dry_run:
        log(f"[DRY ENTRY] {symbol} {'MARKET' if use_market else f'LIMIT@{entry:.8f}'} qty {qty:.8f}")
        log(f"[DRY OCO]   {symbol} SELL qty {qty1:.8f} TP {t1:.8f} SL {stop:.8f}")
        log(f"[DRY TP2]   {symbol} SELL qty {qty2:.8f} LIMIT {t2:.8f}")
        return {"entry_order_id": None, "oco_list_id": None, "t2_order_id": None}

    try:
        if use_market:
            eo = client.order_market_buy(symbol=symbol, quantity=floor_to_step(qty, step_qty))
        else:
            eo = client.order_limit_buy(symbol=symbol, quantity=floor_to_step(qty, step_qty),
                                        price=f"{entry:.8f}", timeInForce="GTC")
        log(f"[LIVE ENTRY] {symbol} qty {qty:.8f} {'market' if use_market else 'limit'}")
    except Exception as e:
        log(f"[ENTRY ERROR] {symbol} {e}")
        time.sleep(1.5)
        try:
            if use_market:
                eo = client.order_market_buy(symbol=symbol, quantity=floor_to_step(qty, step_qty))
            else:
                eo = client.order_limit_buy(symbol=symbol, quantity=floor_to_step(qty, step_qty),
                                            price=f"{entry:.8f}", timeInForce="GTC")
            log(f"[LIVE ENTRY RETRY] {symbol} ok")
        except Exception as e2:
            log(f"[ENTRY FAIL] {symbol} {e2}")
            return None
    entry_order_id = eo.get("orderId")

    try:
        oco = client.create_oco_order(
            symbol=symbol, side="SELL",
            quantity=f"{qty1:.8f}",
            price=f"{t1:.8f}",
            stopPrice=f"{stop:.8f}",
            stopLimitPrice=f"{stop:.8f}",
            stopLimitTimeInForce="GTC"
        )
        oco_list_id = oco.get("orderListId")
        log(f"[LIVE OCO] {symbol} half qty {qty1:.8f} TP1 {t1:.8f} SL {stop:.8f}")
    except Exception as e:
        log(f"[OCO ERROR] {symbol} {e}")
        oco_list_id = None

    try:
        tp2 = client.order_limit_sell(symbol=symbol, quantity=f"{qty2:.8f}", price=f"{t2:.8f}", timeInForce="GTC")
        t2_order_id = tp2.get("orderId")
        log(f"[LIVE TP2] {symbol} half qty {qty2:.8f} @ {t2:.8f}")
    except Exception as e:
        log(f"[TP2 ERROR] {symbol} {e}")
        t2_order_id = None

    return {"entry_order_id": entry_order_id, "oco_list_id": oco_list_id, "t2_order_id": t2_order_id}

def main():
    dry_run = os.getenv("DRY_RUN", "1") != "0"
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_sec = os.getenv("BINANCE_API_SECRET", "")
    client = Client(api_key, api_sec)

    S = fetch_summary()
    if S is None: return

    signals = S.get("signals", {}) or {}
    regime_ok = bool(S.get("regime", {}).get("ok", True))
    orders = S.get("orders", []) or []
    candidates = S.get("candidates", []) or []

    state = load_state()

    try:
        equity_usd, cash_usdt = account_equity_usdt(client)
    except (BinanceAPIException, BinanceRequestException) as e:
        log(f"[ACCOUNT ERROR] {e}")
        equity_usd, cash_usdt = 0.0, 0.0

    gross = current_gross_exposure(state)
    cap = exposure_cap(equity_usd, regime_ok)

    if signals.get("type") == "C":
        for c in candidates[:5]:
            symbol = c["symbol"]; entry = float(c["entry"]); stop = float(c["stop"])
            try:
                min_qty, step_qty, _ = get_symbol_filters(client, symbol)
            except Exception as e:
                log(f"[CAND FILTER] {symbol} {e}"); continue
            qty = size_order(entry, stop, equity_usd, cash_usdt, min_qty, step_qty)
            log(f"[CAND] {symbol} size {qty:.8f} entry {entry:.8f} stop {stop:.8f}")
        return

    if signals.get("type") != "B":
        log("Hold and wait.")
    else:
        for o in orders:
            symbol = o["symbol"]; entry = float(o["entry"]); stop = float(o["stop"]); atr = float(o["atr"])
            prev = state.get(symbol)
            if prev and prev.get("status") != "closed":
                if near(float(prev.get("entry", entry)), entry, DUP_ENTRY_TOL):
                    log(f"[SKIP] {symbol} duplicate near entry"); continue
            last = get_last_price(client, symbol)
            if last > entry + SLIPPAGE_ATR_MULT * atr:
                log(f"[SKIP] {symbol} slippage: {last:.8f} > {entry:.8f} + {SLIPPAGE_ATR_MULT}*ATR"); continue
            try:
                min_qty, step_qty, _ = get_symbol_filters(client, symbol)
            except Exception as e:
                log(f"[FILTER ERR] {symbol} {e}"); continue
            qty = size_order(entry, stop, equity_usd, cash_usdt, min_qty, step_qty)
            if qty <= 0: log(f"[SKIP] {symbol} qty too small or no cash"); continue
            spend = qty * entry
            if gross + spend > cap:
                max_add = max(0.0, cap - gross)
                qty = floor_to_step(max_add / entry, step_qty)
                if qty < min_qty: log(f"[SKIP] {symbol} exposure cap"); continue
                spend = qty * entry
            ids = place_entry_and_exits(client, o, qty, dry_run)
            if ids is None and not dry_run: continue
            state[symbol] = {
                "entry": entry, "stop": stop, "t1": float(o["t1"]), "t2": float(o["t2"]),
                "atr": atr, "filled_qty": qty, "t1_filled_qty": 0.0,
                "t2_order_id": None if ids is None else ids.get("t2_order_id"),
                "oco_list_id": None if ids is None else ids.get("oco_list_id"),
                "last_trail_update_ts": None, "status": "live"
            }
            gross += spend

    for symbol in list(state.keys()):
        try:
            assume_t1_if_hit(client, state, symbol, dry_run)
            manage_trailing(client, state, symbol, dry_run)
        except Exception as e:
            log(f"[MGR ERROR] {symbol} {e}")

    save_state(state)

if __name__ == "__main__":
    main()
