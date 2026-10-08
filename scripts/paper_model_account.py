#!/usr/bin/env python3
"""
paper_model_account.py — the ML model's public paper-trading account.

Run by the daily retrain right after the pipeline (.github/workflows/daily-retrain.yml):
  1. fill the order placed last run, at the next session's open (backend/paper.py rules)
  2. mark the account to the latest close, next to a buy-and-hold benchmark
  3. turn today's prediction into a target for the next open:
     signal "long" → fully invested in the ticker; "short"/"neutral" → all cash

Long-only and all-in/all-out so the track record reads plainly. The ledger,
paper/<TICKER>_model_account.json, is published with the models and served by
GET /api/paper/model.

Usage: python scripts/paper_model_account.py [TICKER]
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import paper      # noqa: E402
import predictor  # noqa: E402

TICKER = (sys.argv[1] if len(sys.argv) > 1 else "NVDA").upper()
LEDGER = ROOT / "paper" / f"{TICKER}_model_account.json"
START_CASH = 100_000.0


def main():
    acct = json.loads(LEDGER.read_text()) if LEDGER.exists() else {
        "ticker": TICKER, "start_cash": START_CASH, "cash": START_CASH, "shares": 0,
        "pending": None, "last_signal": None, "trades": [], "equity": [],
        "benchmark_start_close": None,
    }
    import yfinance as yf
    bars = yf.Ticker(TICKER).history(period="3mo", interval="1d", auto_adjust=False)

    # 1. Fill last run's order at the first session after it was placed
    p = acct["pending"]
    if p:
        got = paper.fill_order({"side": p["side"], "type": "market",
                                "placed_at": pd.Timestamp(p["placed_at"])}, bars)
        if got:
            px, ts = got
            if p["side"] == "buy":
                qty = math.floor(acct["cash"] / px)
                acct["cash"] -= qty * px
                acct["shares"] += qty
            else:
                qty = acct["shares"]
                acct["cash"] += qty * px
                acct["shares"] = 0
            acct["trades"].append({"date": ts.date().isoformat(), "side": p["side"],
                                   "qty": qty, "price": round(px, 4),
                                   "signal": p["signal"],
                                   "predicted_5d_return": p.get("predicted_5d_return")})
            acct["pending"] = None
            print(f"filled {p['side']} {qty} {TICKER} @ {px:.2f} on {ts.date()}")

    # 2. Mark to the latest close
    last_date, close = bars.index[-1].date().isoformat(), float(bars["Close"].iloc[-1])
    if acct["benchmark_start_close"] is None:
        acct["benchmark_start_close"] = close
    point = {"date": last_date, "close": round(close, 4),
             "equity": round(acct["cash"] + acct["shares"] * close, 2),
             "benchmark": round(acct["start_cash"] * close / acct["benchmark_start_close"], 2)}
    acct["equity"] = [e for e in acct["equity"] if e["date"] != last_date] + [point]

    # 3. Today's prediction → target position for the next open
    pred = predictor.predict_latest(TICKER)
    want_long = pred.get("signal") == "long"
    holding = acct["shares"] > 0
    acct["last_signal"] = {"date": pred["date"], "signal": pred.get("signal"),
                           "confidence": pred.get("confidence"),
                           "predicted_5d_return": pred["predictions"].get("predicted_5d_return")}
    if want_long == holding:
        acct["pending"] = None          # a not-yet-filled order the signal no longer wants
    elif acct["pending"] is None:
        acct["pending"] = {
            "side": "buy" if want_long else "sell",
            # decided on the prediction date's close; fills at the next open
            "placed_at": pd.Timestamp(f"{pred['date']} 16:00", tz="America/New_York").isoformat(),
            "signal": pred.get("signal"),
            "predicted_5d_return": pred["predictions"].get("predicted_5d_return"),
        }

    LEDGER.parent.mkdir(exist_ok=True)
    LEDGER.write_text(json.dumps(acct, indent=2))
    print(f"{TICKER} model account: equity {point['equity']:,.2f} vs buy-and-hold "
          f"{point['benchmark']:,.2f}; signal {pred.get('signal')}; pending {acct['pending']}")


if __name__ == "__main__":
    main()
