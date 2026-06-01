"""
signal_narrator.py — LLM-generated prose summaries of ML predictions.

Converts raw output from predictor.predict_latest() into an investor-readable
paragraph that can be surfaced directly in the UI.

Usage:
    from ai.signal_narrator import narrate_predictions
    text = narrate_predictions("NVDA", prediction_dict, sentiment_score=62.0)
"""

from __future__ import annotations

from typing import Optional
from ai.llm_router import chat_completion

_SYSTEM = (
    "You are a quantitative analyst at a hedge fund. "
    "Given ML model predictions and market data, write a concise, professional "
    "signal brief (3–5 sentences). Be factual and precise; avoid hype. "
    "Use percentage figures where given. Never make guarantees about future performance."
)


def _regime_label(regime: Optional[int]) -> str:
    labels = {0: "bear", 1: "neutral", 2: "bull"}
    return labels.get(regime, "unknown") if regime is not None else "unknown"


def _dir_label(direction: Optional[int]) -> str:
    return {0: "downward", 1: "upward"}.get(direction, "flat") if direction is not None else "flat"


def narrate_predictions(
    ticker: str,
    predictions: dict,
    *,
    sentiment_score: Optional[float] = None,
    current_price: Optional[float] = None,
) -> Optional[str]:
    """
    Generate a prose signal brief for `ticker`.

    Parameters
    ----------
    ticker          : stock symbol, e.g. "NVDA"
    predictions     : dict from predictor.predict_latest()
    sentiment_score : 0-100 composite sentiment score (optional)
    current_price   : latest price in USD (optional)

    Returns
    -------
    str  — LLM-generated narrative, or None if the LLM is unavailable
    """
    preds = predictions.get("predictions", {})

    ret1d  = preds.get("predicted_1d_return")
    ret5d  = preds.get("predicted_5d_return")
    vol5d  = preds.get("predicted_vol_5d")
    regime = preds.get("predicted_regime")
    dir1d  = preds.get("predicted_dir_1d")
    signal = predictions.get("signal", "neutral")
    conf   = predictions.get("confidence", "medium")
    anom   = predictions.get("anomaly_flag", 0)
    date   = predictions.get("date", "")

    facts: list[str] = [f"Ticker: {ticker}", f"As of: {date}"]

    if current_price is not None:
        facts.append(f"Current price: ${current_price:.2f}")
    if ret1d is not None:
        facts.append(f"Predicted 1-day return: {ret1d*100:+.2f}%")
    if ret5d is not None:
        facts.append(f"Predicted 5-day return: {ret5d*100:+.2f}%")
    if vol5d is not None:
        facts.append(f"Predicted 5-day realised volatility: {vol5d*100:.2f}%")
    if regime is not None:
        facts.append(f"Market regime: {_regime_label(regime)}")
    if dir1d is not None:
        facts.append(f"Predicted 1-day direction: {_dir_label(dir1d)}")
    facts.append(f"Composite signal: {signal} (confidence: {conf})")
    if anom:
        facts.append("⚠ Anomaly detected — feature values are outside training distribution.")
    if sentiment_score is not None:
        label = "positive" if sentiment_score > 60 else ("negative" if sentiment_score < 40 else "neutral")
        facts.append(f"News sentiment: {label} (score {sentiment_score:.0f}/100)")

    prompt = (
        "Based on the following quantitative signals, write a concise signal brief:\n\n"
        + "\n".join(f"- {f}" for f in facts)
        + "\n\nBrief:"
    )

    return chat_completion(
        [{"role": "user", "content": prompt}],
        system=_SYSTEM,
        max_tokens=300,
        temperature=0.25,
    )
