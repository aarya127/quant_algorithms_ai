"""
tests/test_paper.py

Paper-trading fill rules (backend/paper.py) and the fill endpoint's validation,
with market data stubbed out — no network.
"""
import pandas as pd
import pytest

import paper
from paper import SLIPPAGE, fill_order


def _bars(*rows, start="2026-10-05 09:30", freq="5min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="America/New_York")
    return pd.DataFrame(rows, columns=["Open", "High", "Low"], index=idx)


def _order(side="buy", type_="market", limit=None, at="2026-10-05 09:32"):
    return {"side": side, "type": type_, "limit": limit,
            "placed_at": pd.Timestamp(at, tz="America/New_York")}


class TestFillRules:
    BARS = _bars((100, 101, 99), (102, 103, 101), (104, 105, 98))   # 09:30, 09:35, 09:40

    def test_market_uses_first_bar_starting_after_the_order(self):
        """The 09:30 bar was already trading when the order came in at 09:32."""
        px, ts = fill_order(_order(), self.BARS)
        assert ts == self.BARS.index[1]
        assert px == pytest.approx(102 * (1 + SLIPPAGE))

    def test_market_sell_slips_down(self):
        px, _ = fill_order(_order(side="sell"), self.BARS)
        assert px == pytest.approx(102 * (1 - SLIPPAGE))

    def test_no_bar_after_the_order_means_open(self):
        assert fill_order(_order(at="2026-10-05 10:00"), self.BARS) is None

    def test_limit_buy_fills_at_limit_when_traded_through(self):
        px, ts = fill_order(_order(type_="limit", limit=99.5), self.BARS)
        assert (px, ts) == (99.5, self.BARS.index[2])

    def test_limit_buy_gets_open_when_it_gaps_below(self):
        bars = _bars((100, 101, 99), (95, 96, 94))
        px, _ = fill_order(_order(type_="limit", limit=97), bars)
        assert px == 95

    def test_limit_sell_gets_open_when_it_gaps_above(self):
        bars = _bars((100, 101, 99), (110, 111, 109))
        px, _ = fill_order(_order(side="sell", type_="limit", limit=105), bars)
        assert px == 110

    def test_limit_not_reached_stays_open(self):
        assert fill_order(_order(type_="limit", limit=90), self.BARS) is None

    def test_daily_bars_fill_at_next_session_open(self):
        """An order at Monday's close fills at Tuesday's open (the model account)."""
        days = _bars((100, 101, 99), (103, 104, 102), start="2026-10-05", freq="B")
        px, ts = fill_order(_order(at="2026-10-05 16:00"), days)
        assert ts.date() == pd.Timestamp("2026-10-06").date()
        assert px == pytest.approx(103 * (1 + SLIPPAGE))


@pytest.fixture()
def client(monkeypatch):
    import app as app_module
    monkeypatch.setattr(paper, "quote", lambda s: {
        "price": 100.0, "prev_close": 99.0, "as_of": "x",
        **({"currency": "CAD", "fx_usd": 0.73} if s.endswith(".TO") else
           {"currency": "USD", "fx_usd": 1.0})})
    monkeypatch.setattr(paper, "bars_since", lambda s, since: TestFillRules.BARS)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


class TestFillEndpoint:
    def _post(self, client, **o):
        base = {"id": "a", "symbol": "NVDA", "side": "buy", "qty": 1, "type": "market",
                "placed_at": "2026-10-05T13:32:00Z"}
        r = client.post("/api/paper/fill", json={"orders": [{**base, **o}]})
        return r.get_json()["results"][0]

    def test_fills(self, client):
        res = self._post(client)
        assert res["status"] == "filled" and res["price"] == pytest.approx(102.051)

    @pytest.mark.parametrize("bad, reason", [
        ({"symbol": "../etc"}, "invalid symbol"),
        ({"side": "short"}, "invalid side or type"),
        ({"qty": 0}, "qty or limit out of range"),
        ({"type": "limit", "limit": -5}, "qty or limit out of range"),
        ({"placed_at": "2099-01-01T00:00:00Z"}, "placed_at is in the future"),
    ])
    def test_rejections(self, client, bad, reason):
        res = self._post(client, **bad)
        assert res["status"] == "rejected" and res["reason"] == reason

    def test_foreign_listing_fills_with_its_usd_rate(self, client):
        res = self._post(client, symbol="TD.TO")
        assert res["status"] == "filled"
        assert (res["currency"], res["fx_usd"]) == ("CAD", 0.73)

    def test_too_many_orders(self, client):
        r = client.post("/api/paper/fill", json={"orders": [{}] * 51})
        assert r.status_code == 400
