"""
llm_judge.py — LLM-as-judge evaluation hook for the supervised pipeline.

Called at the end of orchestrator.py after the supervised step completes.
Reads the latest holdout_results.csv, formats key metrics, and asks the
configured LLM to produce a qualitative verdict on whether the retrained
models are worth promoting to production.

The verdict is written to:
    algorithms/machine_learning_algorithms/supervised/output/monitoring/llm_judge_<ticker>.json

Usage (standalone):
    python ai/llm_judge.py NVDA

Called from orchestrator:
    from ai.llm_judge import run_judge
    run_judge(ticker)
"""

from __future__ import annotations

import json
import sys
import datetime
from pathlib import Path

# Project root = ai/../
_PROJECT = Path(__file__).resolve().parent.parent
_OUTPUT  = _PROJECT / "algorithms" / "machine_learning_algorithms" / "supervised" / "output"
_MONITOR = _OUTPUT / "monitoring"

sys.path.insert(0, str(_PROJECT))
from ai_platform.llm_router import chat_completion, active_provider  # noqa: E402


_SYSTEM = (
    "You are a senior quantitative researcher reviewing an automated ML retraining report. "
    "Evaluate whether the retrained models should be promoted to production. "
    "Be concise (≤5 sentences). Focus on: metric quality, risk of overfitting, "
    "anomalies, and a clear promote/hold/reject verdict."
)


def _load_holdout(ticker: str) -> list:
    """Load the most recent holdout results CSV for ticker as a condensed list."""
    try:
        import pandas as pd
        path = _OUTPUT / "holdout_results.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        if "ticker" in df.columns:
            df = df[df["ticker"].str.upper() == ticker.upper()]
        keep = [c for c in df.columns if any(k in c for k in
                ["target", "model", "ic", "f1", "roc", "r2", "mae"])]
        return df[keep].head(20).to_dict(orient="records")
    except Exception:
        return []


def _load_cv(ticker: str) -> list:
    """Load a condensed version of cv_all_results.csv."""
    try:
        import pandas as pd
        path = _OUTPUT / "cv_all_results.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        if "ticker" in df.columns:
            df = df[df["ticker"].str.upper() == ticker.upper()]
        keep = [c for c in df.columns if any(k in c for k in
                ["target", "model", "fold", "ic", "f1", "roc", "r2", "mae"])]
        return df[keep].head(30).to_dict(orient="records")
    except Exception:
        return []


def run_judge(ticker: str) -> dict:
    """
    Run the LLM judge and persist results.

    Returns a dict with keys: ticker, verdict, rationale, timestamp, provider.
    """
    _MONITOR.mkdir(parents=True, exist_ok=True)

    holdout = _load_holdout(ticker)
    cv      = _load_cv(ticker)

    prompt = (
        f"Ticker: {ticker}\n\n"
        "Holdout results (best model per target):\n"
        f"{json.dumps(holdout, indent=2)}\n\n"
        "Cross-validation summary (sample rows):\n"
        f"{json.dumps(cv, indent=2)}\n\n"
        "Based on these results, provide:\n"
        "1. VERDICT: promote | hold | reject\n"
        "2. RATIONALE: 3-4 sentences explaining your verdict.\n"
        "3. RISKS: any specific concerns (overfitting, low IC, class imbalance, etc.)."
    )

    prov = active_provider()
    raw  = chat_completion(
        [{"role": "user", "content": prompt}],
        system=_SYSTEM,
        max_tokens=400,
        temperature=0.1,
    )

    verdict = "unknown"
    if raw:
        lower = raw.lower()
        if "promote" in lower:
            verdict = "promote"
        elif "reject" in lower:
            verdict = "reject"
        elif "hold" in lower:
            verdict = "hold"

    result = {
        "ticker":    ticker,
        "verdict":   verdict,
        "rationale": raw or "LLM unavailable — no verdict produced.",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "provider":  prov,
    }

    out_path = _MONITOR / f"llm_judge_{ticker}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"LOG:LLM judge verdict for {ticker}: {verdict}", flush=True)
    return result


if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    r = run_judge(ticker)
    print(json.dumps(r, indent=2))
