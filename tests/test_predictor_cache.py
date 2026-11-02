"""
tests/test_predictor_cache.py

backend/predictor.py: predict_latest is cached until the features CSV or the
registry index changes, and prediction_log.csv survives concurrent writers.
"""
import threading

import pandas as pd
import pytest

import predictor


@pytest.fixture()
def fake_model(monkeypatch, tmp_path):
    calls = []

    def uncached(ticker):
        calls.append(ticker)
        return {"ticker": ticker, "date": "2026-10-09", "predictions": {"predicted_5d_return": 0.02}}

    monkeypatch.setattr(predictor, "_predict_latest_uncached", uncached)
    monkeypatch.setattr(predictor.model_sync, "ensure_fresh", lambda: None)
    monkeypatch.setattr(predictor, "_PIPELINES", tmp_path)
    monkeypatch.setattr(predictor, "_REGISTRY", tmp_path / "reg")
    predictor._PRED_CACHE.clear()
    (tmp_path / "NVDA_features_with_regimes.csv").write_text("x")
    yield calls, tmp_path
    predictor._PRED_CACHE.clear()


def test_repeat_calls_hit_the_cache(fake_model):
    calls, _ = fake_model
    a = predictor.predict_latest("NVDA")
    b = predictor.predict_latest("NVDA")
    assert calls == ["NVDA"] and a == b


def test_callers_get_copies(fake_model):
    predictor.predict_latest("NVDA")["predictions"]["predicted_5d_return"] = 99
    assert predictor.predict_latest("NVDA")["predictions"]["predicted_5d_return"] == 0.02


def test_new_features_file_invalidates(fake_model):
    import os
    calls, tmp = fake_model
    predictor.predict_latest("NVDA")
    f = tmp / "NVDA_features_with_regimes.csv"
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    predictor.predict_latest("NVDA")
    assert calls == ["NVDA", "NVDA"] and len(predictor._PRED_CACHE) == 1


def test_concurrent_log_writes_leave_a_readable_log(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, "_MONITORING", tmp_path)
    monkeypatch.setattr(predictor, "_PRED_LOG", tmp_path / "prediction_log.csv")
    threads = [threading.Thread(target=predictor._log_prediction,
                                args=({"date": f"2026-10-{d:02d}", "ticker": t, "predictions": {}},))
               for d in range(1, 11) for t in ("NVDA", "AAPL")]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    log = pd.read_csv(tmp_path / "prediction_log.csv")
    assert len(log) == 20 and not log.duplicated(["date", "ticker"]).any()


def test_corrupt_log_is_replaced_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, "_MONITORING", tmp_path)
    monkeypatch.setattr(predictor, "_PRED_LOG", tmp_path / "prediction_log.csv")
    (tmp_path / "prediction_log.csv").write_bytes(b'"unterminated\n\x00,,"\n')
    predictor._log_prediction({"date": "2026-10-09", "ticker": "NVDA", "predictions": {}})
    assert pd.read_csv(tmp_path / "prediction_log.csv")["ticker"].tolist() == ["NVDA"]
