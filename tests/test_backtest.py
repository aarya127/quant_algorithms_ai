"""
tests/test_backtest.py

The backtest simulator (backtesting/simulator.py): decide at the close, fill at
the next open with slippage, re-size from equity like the model paper account,
and never let a later bar change an earlier result. Plus the walk-forward
schedule's rule that a model only trains on labels it could have known.
"""
import math

import numpy as np
import pandas as pd
import pytest

from simulator import (simulate, summarize, max_drawdown, retrain_blocks,
                       signal_from_forecast)

SLIP = 0.0005


def _bars(*oc):
    """Daily bars from (open, close) pairs."""
    return pd.DataFrame({"Date": pd.bdate_range("2026-01-05", periods=len(oc)),
                         "Open": [o for o, _ in oc], "Close": [c for _, c in oc]})


class TestFills:
    BARS = _bars((100, 101), (102, 104), (103, 99), (98, 97))

    def test_decision_fills_at_next_open_with_slippage(self):
        daily, trades = simulate(self.BARS, [1, np.nan, np.nan, np.nan], 10_000, SLIP)
        assert trades[0]["date"] == "2026-01-06"
        assert trades[0]["price"] == pytest.approx(102 * (1 + SLIP))
        assert daily["equity"][0] == 10_000          # nothing held on the decision day

    def test_sized_in_whole_shares_from_equity(self):
        daily, trades = simulate(self.BARS, [1, np.nan, np.nan, np.nan], 10_000, SLIP)
        px = 102 * (1 + SLIP)
        qty = math.floor(10_000 / px)
        assert trades[0]["qty"] == qty
        assert daily["equity"][1] == pytest.approx(10_000 - qty * px + qty * 104)

    def test_long_to_short_is_one_trade_sized_from_equity(self):
        daily, trades = simulate(self.BARS, [1, -1, np.nan, np.nan], 10_000, SLIP)
        assert [t["side"] for t in trades] == ["buy", "sell"]
        sell_px = 103 * (1 - SLIP)
        equity_at_fill = daily["cash"][1] + daily["shares"][1] * sell_px
        assert daily["shares"][2] == -math.floor(equity_at_fill / sell_px)

    def test_repeated_or_missing_signal_does_not_trade(self):
        _, trades = simulate(self.BARS, [1, 1, np.nan, 1], 10_000, SLIP)
        assert len(trades) == 1

    def test_cash_target_closes_the_position(self):
        daily, trades = simulate(self.BARS, [1, 0, np.nan, np.nan], 10_000, SLIP)
        assert trades[-1]["side"] == "sell" and daily["shares"][2] == 0

    def test_decision_on_the_last_row_never_fills(self):
        _, trades = simulate(self.BARS, [np.nan, np.nan, np.nan, 1], 10_000, SLIP)
        assert trades == []

    def test_later_bars_cannot_change_earlier_equity(self):
        """No look-ahead: rewriting the future leaves the past untouched."""
        a, _ = simulate(self.BARS, [1, -1, 0, 1], 10_000, SLIP)
        future = self.BARS.copy()
        future.loc[3, ["Open", "Close"]] = [500, 10]
        b, _ = simulate(future, [1, -1, 0, 1], 10_000, SLIP)
        assert a["equity"][:3].tolist() == b["equity"][:3].tolist()

    def test_wiped_out_account_cannot_reopen(self):
        """Short through a tripling, cover, then ask to short again with < 0 equity."""
        bars = _bars((100, 100), (100, 300), (300, 300), (300, 300))
        daily, trades = simulate(bars, [-1, 0, -1, np.nan], 10_000, 0.0)
        assert daily["equity"][2] < 0
        assert daily["shares"][3] == 0 and len(trades) == 2

    def test_rejects_misaligned_targets(self):
        with pytest.raises(ValueError):
            simulate(self.BARS, [1, 0], 10_000, SLIP)


class TestMetrics:
    def test_max_drawdown(self):
        assert max_drawdown([100, 120, 90, 130, 65]) == pytest.approx(65 / 130 - 1)
        assert max_drawdown([100, 110, 120]) == 0.0

    def test_summary_of_a_flat_account(self):
        daily = pd.DataFrame({"Date": pd.bdate_range("2026-01-05", periods=3),
                              "shares": [0, 0, 0], "equity": [1000.0] * 3})
        s = summarize(daily, [], 1000.0)
        assert s["total_return"] == 0 and s["sharpe"] == 0 and s["exposure"] == 0

    def test_summary_returns_and_exposure(self):
        daily = pd.DataFrame({"Date": pd.bdate_range("2026-01-05", periods=2),
                              "shares": [10, -10], "equity": [1100.0, 1210.0]})
        s = summarize(daily, [{}, {}], 1000.0)
        assert s["total_return"] == pytest.approx(0.21)
        assert s["long_frac"] == 0.5 and s["short_frac"] == 0.5 and s["n_trades"] == 2


class TestWalkForward:
    @pytest.mark.parametrize("horizon", [1, 5])
    def test_trains_only_on_labels_known_before_predicting(self, horizon):
        for train_stop, s, _ in retrain_blocks(300, 100, 21, horizon):
            last_label_row = train_stop - 1
            assert last_label_row + horizon <= s - 1     # label complete by s-1's close

    def test_blocks_cover_every_row_once(self):
        blocks = retrain_blocks(300, 100, 21, 5)
        covered = [r for _, s, e in blocks for r in range(s, e)]
        assert covered == list(range(100, 300))

    def test_signal_rule(self):
        out = signal_from_forecast([0.02, -0.02, 0.005, np.nan, 0.01], 0.01)
        assert out[:3].tolist() == [1, -1, 0] and np.isnan(out[3]) and out[4] == 0
