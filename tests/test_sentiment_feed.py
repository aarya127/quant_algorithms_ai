"""
tests/test_sentiment_feed.py

The Sentiment tab's data layer (backend/sentiment_feed.py) and route, with the
Polygon and Finnhub calls stubbed — no network.
"""
import datetime

import pytest

import sentiment_feed as sf

TODAY = datetime.date(2026, 11, 2)


def _poly(ticker="NVDA", sentiment="positive", when="2026-11-01T14:30:00Z", title="Headline",
          others=(), reason="Strong demand"):
    ins = [{"ticker": t, "sentiment": "positive", "sentiment_reasoning": "other"} for t in others]
    if sentiment is not None:
        ins.append({"ticker": ticker, "sentiment": sentiment, "sentiment_reasoning": reason})
    return {"published_utc": when, "title": title, "article_url": "https://x.test/a",
            "publisher": {"name": "Wire"}, "insights": ins}


def _fh(title, ts=datetime.datetime(2026, 11, 1, 15, tzinfo=datetime.timezone.utc)):
    return {"datetime": int(ts.timestamp()), "headline": title, "source": "Reuters", "url": "https://x.test/f"}


@pytest.fixture(autouse=True)
def _fresh_state():
    sf._cache.clear()
    yield
    sf._cache.clear()


class TestParsing:
    @pytest.mark.parametrize("score, label", [(0.15, "positive"), (0.149, "neutral"),
                                              (-0.149, "neutral"), (-0.15, "negative"),
                                              (None, "unscored")])
    def test_labels_use_the_neutral_band(self, score, label):
        assert sf.label_for(score) == label

    def test_uses_the_requested_tickers_view_and_reason(self):
        [a] = sf.parse_polygon([_poly(sentiment="negative", others=("AMD",), reason="Export curbs")], "NVDA")
        assert a["score"] == -1.0 and a["label"] == "negative" and a["reason"] == "Export curbs"
        assert a["source"] == "Wire" and a["url"]

    def test_article_without_this_tickers_insight_stays_unscored(self):
        [a] = sf.parse_polygon([_poly(sentiment=None, others=("AMD",))], "NVDA")
        assert a["score"] is None and a["label"] == "unscored" and a["reason"] == ""

    def test_bad_timestamps_are_skipped_and_newest_comes_first(self):
        out = sf.parse_polygon([_poly(when="2026-10-30T00:00:00Z", title="old"), _poly(when="garbage"),
                                _poly(when="2026-11-01T00:00:00Z", title="new")], "NVDA")
        assert [a["headline"] for a in out] == ["new", "old"]

    def test_finnhub_drops_items_before_the_window_and_bad_timestamps(self):
        old = _fh("old", datetime.datetime(2026, 9, 1, tzinfo=datetime.timezone.utc))
        out = sf.parse_finnhub([_fh("fresh"), old, {"datetime": "x", "headline": "bad"}], "2026-10-03")
        assert [a["headline"] for a in out] == ["fresh"] and out[0]["score"] is None

    def test_merge_prefers_the_scored_copy_of_a_headline(self):
        scored = sf.parse_polygon([_poly(title="Nvidia beats estimates")], "NVDA")
        merged = sf.merge(scored, sf.parse_finnhub([_fh("NVIDIA  beats estimates"), _fh("Other")], "2026-10-01"))
        assert [a["provider"] for a in merged].count("polygon") == 1 and len(merged) == 2


class TestAggregation:
    def test_no_scored_articles_means_no_score_not_neutral_zero(self):
        s = sf.summarize(sf.parse_finnhub([_fh("a"), _fh("b")], "2026-10-01"))
        assert s["overall_score"] is None and s["overall_label"] == "no_data"
        assert s["unscored_total"] == 2

    def test_overall_score_is_net_share_of_positive_over_negative(self):
        arts = sf.parse_polygon([_poly(sentiment=s, title=str(i)) for i, s in
                                 enumerate(["positive", "positive", "neutral", "negative"])], "NVDA")
        s = sf.summarize(arts)
        assert s["overall_score"] == pytest.approx(0.25) and s["overall_label"] == "positive"
        assert (s["positive_count"], s["neutral_count"], s["negative_count"]) == (2, 1, 1)

    def test_daily_series_counts_every_article_but_scores_only_scored_ones(self):
        arts = sf.merge(sf.parse_polygon([_poly(sentiment="negative")], "NVDA"),
                        sf.parse_finnhub([_fh("x"), _fh("y", datetime.datetime(2026, 10, 30, tzinfo=datetime.timezone.utc))],
                                         "2026-10-01"))
        series = {d["date"]: d for d in sf.daily_series(arts)}
        assert series["2026-11-01"] == {"date": "2026-11-01", "score": -1.0, "count": 2}
        assert series["2026-10-30"]["score"] is None

    def test_capped_feed_reports_its_real_coverage(self):
        feed = [_poly(when="2026-11-01T00:00:00Z", title=f"t{i}") for i in range(sf.POLYGON_LIMIT)]
        p = sf.build_payload("NVDA", 90, "2026-08-04", feed, "ok", [])
        assert p["coverage_start"] == "2026-11-01" and any("at most" in n for n in p["notes"])
        assert len(p["articles"]) == sf.MAX_ARTICLES and p["scored_total"] == sf.POLYGON_LIMIT

    @pytest.mark.parametrize("status, phrase", [("rate_limited", "5 calls a minute"), ("auth", "rejected"),
                                                ("no_key", "POLYGON_API_KEY"), ("error", "didn't respond")])
    def test_degraded_polygon_is_explained(self, status, phrase):
        p = sf.build_payload("NVDA", 30, "2026-10-03", [], status, [_fh("x")])
        assert any(phrase in n for n in p["notes"]) and p["overall_score"] is None


class FakeNews:
    def __init__(self, results, status="ok"):
        self.results, self.status, self.calls = results, status, []

    def __call__(self, symbol, start):
        self.calls.append((symbol, start))
        return self.results, self.status


class FakeResp:
    def __init__(self, code, body):
        self.status_code, self._body = code, body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class TestFetching:
    def test_one_polygon_call_then_cached_and_finnhub_skipped(self):
        news, fh_calls = FakeNews([_poly()]), []
        a = sf.get_feed("NVDA", 30, lambda *a: fh_calls.append(a) or [_fh("x")], news, today=TODAY)
        b = sf.get_feed("NVDA", 30, lambda *a: [], news, today=TODAY)
        assert a is b and news.calls == [("NVDA", "2026-10-03")] and fh_calls == []
        assert a["scored_total"] == 1 and a["as_of"]

    def test_polygon_failure_falls_back_to_finnhub_and_is_not_cached(self):
        news = FakeNews([], "rate_limited")
        a = sf.get_feed("NVDA", 30, lambda *a: [_fh("x")], news, today=TODAY)
        news.results, news.status = [_poly()], "ok"
        b = sf.get_feed("NVDA", 30, lambda *a: [], news, today=TODAY)
        assert a["unscored_total"] == 1 and a["notes"] and b["scored_total"] == 1 and len(news.calls) == 2

    def test_finnhub_failure_still_returns_a_payload(self):
        def boom(*a):
            raise RuntimeError("finnhub down")
        p = sf.get_feed("NVDA", 30, boom, FakeNews([], "error"), today=TODAY)
        assert p["articles_total"] == 0 and p["notes"]

    @pytest.mark.parametrize("resp, status", [(FakeResp(200, {"results": [_poly()]}), "ok"),
                                              (FakeResp(200, {"status": "OK"}), "ok"),
                                              (FakeResp(401, {}), "auth"), (FakeResp(429, {}), "rate_limited"),
                                              (FakeResp(500, {}), "error"), (FakeResp(200, ValueError()), "error")])
    def test_polygon_replies(self, monkeypatch, resp, status):
        monkeypatch.setattr(sf.requests, "get", lambda *a, **k: resp)
        assert sf.polygon_news("NVDA", "2026-10-03", key="k")[1] == status

    def test_missing_key_makes_no_call(self, monkeypatch):
        def no_call(*a, **k):
            raise AssertionError("called")
        monkeypatch.setattr(sf.requests, "get", no_call)
        assert sf.polygon_news("NVDA", "2026-10-03", key="") == ([], "no_key")


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
