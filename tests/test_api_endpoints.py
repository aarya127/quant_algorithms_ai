"""
tests/test_api_endpoints.py

Flask test_client tests for backend/app.py — no network, no keys, no subprocesses.

Covers the contract of the always-available endpoints (health, pipeline job
lifecycle incl. auth + single-flight) and input validation on the chart/news
routes (bad params must 4xx or degrade gracefully, never uncaught-500).

Heavy paths are cut off at the module boundary:
  * PIPELINE_DB_PATH points at a temp SQLite file (set before import),
  * _run_retrain_job is monkeypatched so no orchestrator subprocess ever spawns,
  * data-layer functions are monkeypatched where a route would call outward.
"""
import os
import sys
import tempfile

# --- env must be set BEFORE app/pipeline_store import (both read env at import) ---
_TMPDIR = tempfile.mkdtemp(prefix="api_tests_")
os.environ["PIPELINE_DB_PATH"] = os.path.join(_TMPDIR, "pipeline_state.db")
os.environ["PIPELINE_TRIGGER_TOKEN"] = "test-token-123"

import pytest  # noqa: E402

pytest.importorskip("flask", reason="flask not installed — backend deps required")
pytest.importorskip("cachetools", reason="cachetools not installed — backend deps required")

# conftest.py puts backend/ on sys.path
import app as app_module          # noqa: E402
import pipeline_store             # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    # Never spawn the real orchestrator from tests.
    monkeypatch.setattr(app_module, "_run_retrain_job", lambda job_id, ticker: None)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def _clean_jobs():
    """Each test starts with an empty job table."""
    pipeline_store.init_db()
    import sqlite3
    sqlite3.connect(pipeline_store._DB_PATH).execute("DELETE FROM jobs").connection.commit()
    yield


# --- health -----------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


# --- pipeline trigger: auth -------------------------------------------------

def test_pipeline_run_requires_token(client):
    r = client.post("/api/pipeline/run", json={"ticker": "NVDA"})
    assert r.status_code == 401
    assert r.get_json()["success"] is False


def test_pipeline_run_rejects_wrong_token(client):
    r = client.post("/api/pipeline/run", json={"ticker": "NVDA"},
                    headers={"X-Pipeline-Token": "nope"})
    assert r.status_code == 401


def test_pipeline_run_accepts_valid_token(client):
    r = client.post("/api/pipeline/run", json={"ticker": "NVDA"},
                    headers={"X-Pipeline-Token": "test-token-123"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True
    assert body["ticker"] == "NVDA"
    assert body["status"] == "queued"
    assert len(body["job_id"]) == 8


# --- pipeline trigger: single-flight -----------------------------------------

def test_pipeline_run_single_flight_409(client):
    hdr = {"X-Pipeline-Token": "test-token-123"}
    first = client.post("/api/pipeline/run", json={"ticker": "NVDA"}, headers=hdr)
    assert first.status_code == 200
    job_id = first.get_json()["job_id"]

    second = client.post("/api/pipeline/run", json={"ticker": "AMD"}, headers=hdr)
    assert second.status_code == 409
    body = second.get_json()
    assert body["job_id"] == job_id  # points at the active job

    # once the job finishes, a new trigger is allowed again
    pipeline_store.set_status(job_id, status="done")
    third = client.post("/api/pipeline/run", json={"ticker": "AMD"}, headers=hdr)
    assert third.status_code == 200


# --- pipeline status ----------------------------------------------------------

def test_pipeline_status_unknown_404(client):
    r = client.get("/api/pipeline/status/nonexist")
    assert r.status_code == 404
    assert r.get_json()["success"] is False


def test_pipeline_status_roundtrip(client):
    hdr = {"X-Pipeline-Token": "test-token-123"}
    job_id = client.post("/api/pipeline/run", json={"ticker": "NVDA"},
                         headers=hdr).get_json()["job_id"]

    pipeline_store.set_status(job_id, status="running", current_step="supervised")
    pipeline_store.append_log(job_id, "Starting step: supervised")

    r = client.get(f"/api/pipeline/status/{job_id}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "running"
    assert body["current_step"] == "supervised"
    assert body["last_logs"] == ["Starting step: supervised"]


def test_pipeline_status_surfaces_error(client):
    hdr = {"X-Pipeline-Token": "test-token-123"}
    job_id = client.post("/api/pipeline/run", json={"ticker": "NVDA"},
                         headers=hdr).get_json()["job_id"]
    pipeline_store.set_status(job_id, status="error", error="Step 'extract' failed")

    body = client.get(f"/api/pipeline/status/{job_id}").get_json()
    assert body["status"] == "error"
    assert "extract" in body["error"]


# --- chart param validation ---------------------------------------------------

def test_charts_invalid_period_400(client):
    r = client.get("/api/charts/NVDA?period=bogus")
    assert r.status_code == 400
    assert "invalid period" in r.get_json()["error"]


def test_charts_invalid_interval_400(client):
    r = client.get("/api/charts/NVDA?period=1y&interval=17q")
    assert r.status_code == 400
    assert "invalid interval" in r.get_json()["error"]


def test_indicators_invalid_period_400(client):
    r = client.get("/api/charts/NVDA/indicators?period=nope")
    assert r.status_code == 400


def test_compare_requires_symbols_400(client):
    r = client.get("/api/charts/compare")
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_compare_invalid_period_400(client):
    r = client.get("/api/charts/compare?symbols=NVDA,AMD&period=zzz")
    assert r.status_code == 400


# --- news count param robustness ----------------------------------------------
# Bad `count` must not raise an uncaught 500; the routes fall back to the default
# and proceed. The outward calls are stubbed so no network is touched.

def test_twitter_news_bad_count_does_not_500(client, monkeypatch):
    monkeypatch.setattr(app_module, "get_market_tweets",
                        lambda symbol=None, count=20: [], raising=False)
    r = client.get("/api/news/twitter?count=abc")
    assert r.status_code == 200
    assert r.get_json()["success"] is True


def test_alpaca_news_bad_count_does_not_500(client, monkeypatch):
    monkeypatch.setattr(app_module, "get_recent_news",
                        lambda count=20, symbol=None: [], raising=False)
    r = client.get("/api/news/alpaca?count=-5")
    assert r.status_code == 200


def test_twitter_news_count_is_capped(client, monkeypatch):
    seen = {}

    def fake_tweets(symbol=None, count=20):
        seen["count"] = count
        return []

    monkeypatch.setattr(app_module, "get_market_tweets", fake_tweets, raising=False)
    client.get("/api/news/twitter?count=99999")
    assert seen["count"] == 100  # clamped to the max


# --- route table sanity ---------------------------------------------------------

def test_no_duplicate_route_rules():
    """Each (rule, method) pair maps to exactly one endpoint — guards against the
    duplicate-registration bug where a second handler is silently unreachable."""
    seen = {}
    for rule in app_module.app.url_map.iter_rules():
        for method in rule.methods - {"HEAD", "OPTIONS"}:
            key = (rule.rule, method)
            assert key not in seen, f"duplicate route {key}: {seen[key]} vs {rule.endpoint}"
            seen[key] = rule.endpoint
