"""
tests/test_paper_model_account.py

The ML model's paper account (scripts/paper_model_account.py), with the
predictor and market data stubbed: decide at the close, fill at the next open,
long / short / cash from the signal, sized from equity at the fill.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from paper import SLIPPAGE

ROOT = Path(__file__).resolve().parents[1]


class World:
    """Daily bars and the model's signal, advanced one session per run."""
    def __init__(self):
        self.bars = pd.DataFrame(columns=["Open", "High", "Low", "Close"],
                                 index=pd.DatetimeIndex([], tz="America/New_York"))
        self.signal = "neutral"

    def session(self, date, open_, close):
        ts = pd.Timestamp(date, tz="America/New_York")
        self.bars.loc[ts] = [open_, max(open_, close), min(open_, close), close]

    def predict_latest(self, ticker):
        return {"date": self.bars.index[-1].date().isoformat(), "signal": self.signal,
                "confidence": "medium", "predictions": {"predicted_5d_return": 0.01}}


@pytest.fixture()
def account(tmp_path, monkeypatch):
    world = World()
    monkeypatch.setitem(sys.modules, "predictor", types.SimpleNamespace(predict_latest=world.predict_latest))
    yf = types.SimpleNamespace(Ticker=lambda t: types.SimpleNamespace(history=lambda **kw: world.bars))
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    spec = importlib.util.spec_from_file_location("pma", ROOT / "scripts" / "paper_model_account.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.TICKER = "NVDA"
    mod.LEDGER = tmp_path / "NVDA_model_account.json"

    def run(date, open_, close, signal):
        world.session(date, open_, close)
        world.signal = signal
        mod.main()
        return json.loads(mod.LEDGER.read_text())
    run.ledger = mod.LEDGER
    return run


def test_decides_at_close_fills_at_next_open(account):
    a = account("2026-10-05", 99, 100, "long")
    assert a["shares"] == 0 and a["pending"]["target"] == "long"
    a = account("2026-10-06", 100, 104, "long")
    px = 100 * (1 + SLIPPAGE)
    assert a["shares"] == int(100_000 // px) and a["pending"] is None
    assert a["trades"][0]["price"] == pytest.approx(px)


def test_long_to_short_is_one_trade_sized_from_equity(account):
    account("2026-10-05", 99, 100, "long")
    account("2026-10-06", 100, 100, "short")           # long fills; short ordered
    a = account("2026-10-07", 110, 110, "short")       # flips at 110
    long_qty = a["trades"][0]["qty"]
    px = 110 * (1 - SLIPPAGE)
    equity = 100_000 - long_qty * 100 * (1 + SLIPPAGE) + long_qty * px
    assert a["shares"] == -int(equity // px)
    assert a["trades"][1]["side"] == "sell" and a["trades"][1]["qty"] == long_qty + int(equity // px)
    # marked at the 110 close, not the fill price
    assert a["equity"][-1]["equity"] == pytest.approx(a["cash"] + a["shares"] * 110, abs=0.01)


def test_short_profits_when_price_falls_then_covers_to_cash(account):
    account("2026-10-05", 100, 100, "short")
    account("2026-10-06", 100, 100, "neutral")         # short fills at 100
    a = account("2026-10-07", 90, 90, "neutral")       # covers at 90 → cash
    assert a["shares"] == 0
    assert a["cash"] > 100_000 * 1.09                  # ~10% gain on a full-size short


def test_migrates_long_only_ledger(account):
    account("2026-10-05", 100, 100, "neutral")
    led = json.loads(account.ledger.read_text())
    led["pending"] = {"side": "buy", "placed_at": "2026-10-05T16:00:00-04:00",
                      "signal": "long", "predicted_5d_return": 0.02}
    account.ledger.write_text(json.dumps(led))
    a = account("2026-10-06", 100, 100, "long")
    assert a["shares"] > 0 and a["trades"][0]["target"] == "long"
