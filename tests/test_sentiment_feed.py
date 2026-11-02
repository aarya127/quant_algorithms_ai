"""
tests/test_sentiment_feed.py

The Sentiment tab's data layer (backend/sentiment_feed.py) and route, with the
Alpha Vantage and Finnhub clients stubbed — no network.
"""
import datetime

import pytest

import sentiment_feed as sf

TODAY = datetime.date(2026, 11, 2)


def _av(ticker="NVDA", score=0.3, rel=0.8, when="20261101T143000", title="Headline", others=()):
    sents = [{"ticker": t, "ticker_sentiment_score": "0.9", "relevance_score": "0.9"} for t in others]
    sents.append({"ticker": ticker, "ticker_sentiment_score": str(score), "relevance_score": str(rel)})
    return {"time_published": when, "title": title, "url": "https://x.test/a", "source": "Wire",
            "ticker_sentiment": sents}


def _fh(title, ts=datetime.datetime(2026, 11, 1, 15, tzinfo=datetime.timezone.utc)):
    return {"datetime": int(ts.timestamp()), "headline": title, "source": "Reuters", "url": "https://x.test/f"}


@pytest.fixture(autouse=True)
def _fresh_state():
    sf._cache.clear()
    sf._av_used.update(day=None, calls=0)
    yield
    sf._cache.clear()


class TestParsing:
    @pytest.mark.parametrize("score, label", [(0.15, "positive"), (0.149, "neutral"),
                                              (-0.149, "neutral"), (-0.15, "negative"),
                                              (None, "unscored")])
    def test_labels_use_alpha_vantage_neutral_band(self, score, label):
        assert sf.label_for(score) == label

    def test_uses_the_requested_tickers_score_not_another_tickers(self):
        [a] = sf.parse_av_feed([_av(score=-0.4, others=("AMD",))], "NVDA")
        assert a["score"] == -0.4 and a["label"] == "negative" and a["url"]

    def test_skips_articles_without_the_ticker_or_with_bad_values(self):
        feed = [_av(ticker="AMD"), _av(when="garbage"), _av(score="nan"),
                {"time_published": "20261101T000000", "ticker_sentiment": [{"ticker": "NVDA"}]}]
        assert sf.parse_av_feed(feed, "NVDA") == []

    def test_relevance_floor_and_newest_first(self):
        out = sf.parse_av_feed([_av(rel=0.0, when="20261030T000000", title="old"),
                                _av(when="20261101T000000", title="new")], "NVDA")
        assert [a["headline"] for a in out] == ["new", "old"]
        assert out[1]["relevance"] == sf.MIN_RELEVANCE

    def test_finnhub_drops_items_before_the_window_and_bad_timestamps(self):
        old = _fh("old", datetime.datetime(2026, 9, 1, tzinfo=datetime.timezone.utc))
        out = sf.parse_finnhub([_fh("fresh"), old, {"datetime": "x", "headline": "bad"}], "2026-10-03")
        assert [a["headline"] for a in out] == ["fresh"] and out[0]["score"] is None

    def test_merge_prefers_the_scored_copy_of_a_headline(self):
        scored = sf.parse_av_feed([_av(title="Nvidia beats estimates")], "NVDA")
        merged = sf.merge(scored, sf.parse_finnhub([_fh("NVIDIA  beats estimates"), _fh("Other")], "2026-10-01"))
        assert [a["provider"] for a in merged].count("alphavantage") == 1 and len(merged) == 2


class TestAggregation:
    def test_no_scored_articles_means_no_score_not_neutral_zero(self):
        s = sf.summarize(sf.parse_finnhub([_fh("a"), _fh("b")], "2026-10-01"))
        assert s["overall_score"] is None and s["overall_label"] == "no_data"
        assert s["unscored_total"] == 2

    def test_overall_score_is_relevance_weighted(self):
        arts = sf.parse_av_feed([_av(score=0.5, rel=0.9, title="a"),
                                 _av(score=-0.5, rel=0.1, title="b")], "NVDA")
        assert sf.summarize(arts)["overall_score"] == pytest.approx(0.4)

    def test_daily_series_counts_every_article_but_scores_only_scored_ones(self):
        arts = sf.merge(sf.parse_av_feed([_av(score=0.2)], "NVDA"),
                        sf.parse_finnhub([_fh("x"), _fh("y", datetime.datetime(2026, 10, 30, tzinfo=datetime.timezone.utc))],
                                         "2026-10-01"))
        series = {d["date"]: d for d in sf.daily_series(arts)}
        assert series["2026-11-01"] == {"date": "2026-11-01", "score": 0.2, "count": 2}
        assert series["2026-10-30"]["score"] is None

    def test_capped_feed_reports_its_real_coverage(self):
        feed = [_av(when="20261101T000000", title=f"t{i}") for i in range(sf.AV_LIMIT)]
        p = sf.build_payload("NVDA", 90, "2026-08-04", feed, "ok", [])
        assert p["coverage_start"] == "2026-11-01" and any("at most" in n for n in p["notes"])
        assert len(p["articles"]) == sf.MAX_ARTICLES and p["scored_total"] == sf.AV_LIMIT

    @pytest.mark.parametrize("status, phrase", [("rate_limited", "daily limit"),
                                                ("budget", "budget"), ("error", "didn't respond")])
    def test_degraded_alpha_vantage_is_explained(self, status, phrase):
        p = sf.build_payload("NVDA", 30, "2026-10-03", [], status, [_fh("x")])
        assert any(phrase in n for n in p["notes"]) and p["overall_score"] is None


class FakeAV:
    api_key = "k"

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def get_market_news_sentiment(self, **kw):
        self.calls.append(kw)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class TestFetching:
    def test_one_av_call_newest_first_then_cached(self):
        av = FakeAV({"feed": [_av()]})
        a = sf.get_feed("NVDA", 30, av, lambda *a: [_fh("x")], today=TODAY)
        b = sf.get_feed("NVDA", 30, av, lambda *a: [], today=TODAY)
        assert a is b and len(av.calls) == 1
        assert av.calls[0]["sort"] == "LATEST" and av.calls[0]["time_from"] == "20261003T0000"
        assert a["scored_total"] == 1 and a["as_of"]

    @pytest.mark.parametrize("reply, status", [({"Information": "limit"}, "rate_limited"),
                                               ({"Error Message": "bad ticker"}, "ok"),
                                               (RuntimeError("down"), "error"), ("junk", "error")])
    def test_alpha_vantage_replies(self, reply, status):
        assert sf._fetch_av(FakeAV(reply), "NVDA", "2026-10-03", "2026-11-02", "2026-11-02")[1] == status

    def test_missing_key_makes_no_call(self):
        av = FakeAV({"feed": []})
        av.api_key = None
        assert sf._fetch_av(av, "NVDA", "a", "b", "2026-11-02") == ([], "error") and av.calls == []

    def test_daily_budget_stops_alpha_vantage_calls(self):
        av = FakeAV({"feed": []})
        for i in range(sf.AV_DAILY_BUDGET + 2):
            sf.get_feed(f"T{i}", 30, av, lambda *a: [], today=TODAY)
        assert len(av.calls) == sf.AV_DAILY_BUDGET
        assert any("budget" in n for n in sf.get_feed("ZZZ", 30, av, lambda *a: [], today=TODAY)["notes"])

    def test_finnhub_failure_still_returns_scored_news(self):
        def boom(*a):
            raise RuntimeError("finnhub down")
        p = sf.get_feed("NVDA", 30, FakeAV({"feed": [_av()]}), boom, today=TODAY)
        assert p["scored_total"] == 1


class TestRoute:
    @pytest.fixture()
    def client(self):
        import app as app_module
        app_module.app.config["TESTING"] = True
        return app_module.app.test_client()

    def test_bad_days_falls_back_and_symbol_is_upper_cased(self, client, monkeypatch):
        seen = {}
        monkeypatch.setattr(sf, "get_feed", lambda sym, days, *a: seen.update(sym=sym, days=days) or {"x": 1})
        d = client.get("/api/sentiment/news/nvda?days=abc").get_json()
        assert d["success"] and seen == {"sym": "NVDA", "days": 30}

    def test_days_are_capped(self, client, monkeypatch):
        seen = {}
        monkeypatch.setattr(sf, "get_feed", lambda sym, days, *a: seen.update(days=days) or {})
        client.get("/api/sentiment/news/NVDA?days=9999")
        assert seen["days"] == 365

    def test_failure_is_a_500_without_a_traceback(self, client, monkeypatch):
        def boom(*a):
            raise RuntimeError("secret internals")
        monkeypatch.setattr(sf, "get_feed", boom)
        r = client.get("/api/sentiment/news/NVDA")
        assert r.status_code == 500 and "secret" not in r.get_data(as_text=True)
        assert "traceback" not in r.get_json()

    def test_old_multi_source_route_is_gone(self, client):
        assert client.get("/api/sentiment/NVDA").status_code == 404
