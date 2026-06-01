#!/usr/bin/env python3
"""
orchestrator.py — End-to-end ML pipeline orchestrator.

Chains ETL → clean → normalize → unsupervised → supervised in a single run.
Designed to be invoked by the Flask backend (via subprocess) or run directly.

Usage:
    python orchestrator.py [TICKER]          # default: NVDA

Output protocol (parsed by the backend):
    STEP:<name>:start    — step is beginning
    LOG:<text>           — passthrough line from the subprocess
    STEP:<name>:done     — step completed successfully
    STATUS:up_to_date    — data is current; ML steps are skipped
    STATUS:done          — all steps completed; new models may be in the registry
    STATUS:error:<name>  — step <name> failed (non-zero exit code)

Exit codes:
    0  — pipeline finished (done or up_to_date)
    1  — a step failed
"""

import os
import sys
import subprocess
import datetime
from contextlib import nullcontext
from pathlib import Path

TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
PYTHON = sys.executable
ROOT   = Path(__file__).resolve().parents[2]   # quant_algorithms_ai/
PIPES  = ROOT / "algorithms" / "machine_learning_algorithms" / "data_pipelines"
ML     = ROOT / "algorithms" / "machine_learning_algorithms"

STEPS = [
    ("extract",      [PYTHON, str(PIPES / "run_pipeline.py"), TICKER]),
    ("clean",        [PYTHON, str(PIPES / "clean.py"),        TICKER]),
    ("normalize",    [PYTHON, str(PIPES / "normalize.py"),    TICKER]),
    ("unsupervised", [PYTHON, str(ML / "unsupervised" / "unsupervised.py"), TICKER]),
    ("supervised",   [PYTHON, str(ML / "supervised"   / "supervised.py"),   TICKER]),
]

env = os.environ.copy()
# Default to no hyperparameter tuning for daily runs (fast path).
# Override by setting USE_TUNING=1 in the environment before calling.
env.setdefault("USE_TUNING", "0")

# MLflow tracing — best-effort; pipeline continues even if mlflow is absent or broken.
# Each orchestrator run creates one root trace (visible in Observability > Traces)
# with one child span per step so you can see where time is spent.
try:
    import mlflow as _mlflow
    _mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    _mlflow.set_experiment(TICKER)
    _MLFLOW = _mlflow
except Exception:
    _MLFLOW = None


def _run_step(name, cmd):
    """
    Run a pipeline step as a subprocess.

    Returns (returncode, up_to_date).
    Streams LOG: lines for each output line.
    Each step is wrapped in an MLflow child span when tracing is available.
    """
    print(f"STEP:{name}:start", flush=True)
    span_cm = _MLFLOW.start_span(name=name, span_type="TASK") if _MLFLOW else nullcontext()
    with span_cm as span:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
            env=env,
        )
        up_to_date = False
        for raw in proc.stdout:
            line = raw.rstrip()
            print(f"LOG:{line}", flush=True)
            if name == "extract" and "already up to date" in line.lower():
                up_to_date = True
        proc.wait()
        if span is not None:
            try:
                span.set_inputs({"ticker": TICKER, "step": name})
                span.set_outputs({"returncode": proc.returncode, "up_to_date": up_to_date})
            except Exception:
                pass
    return proc.returncode, up_to_date


def _run_judge(ticker: str) -> None:
    """Best-effort LLM-as-judge evaluation; never blocks the pipeline."""
    try:
        import importlib, sys as _sys
        _sys.path.insert(0, str(ROOT))
        judge_mod = importlib.import_module("ai.llm_judge")
        judge_mod.run_judge(ticker)
    except Exception as exc:
        print(f"LOG:LLM judge skipped ({exc})", flush=True)


def _run_pipeline():
    """Execute all steps in order; return (status_str, exit_code)."""
    for step_name, step_cmd in STEPS:
        rc, up_to_date = _run_step(step_name, step_cmd)

        if rc != 0:
            print(f"STATUS:error:{step_name}", flush=True)
            return "error", 1

        print(f"STEP:{step_name}:done", flush=True)

        if up_to_date:
            # Feature data is already current for today.
            # Skip retraining — the existing registry is still valid.
            print("STATUS:up_to_date", flush=True)
            return "up_to_date", 0

    # Run LLM-as-judge after a successful supervised retrain
    _run_judge(TICKER)

    print("STATUS:done", flush=True)
    return "done", 0


_today = datetime.date.today().isoformat()
_root_cm = (
    _MLFLOW.start_span(name=f"pipeline_{TICKER}_{_today}", span_type="CHAIN")
    if _MLFLOW else nullcontext()
)
with _root_cm as _root_span:
    if _root_span is not None:
        try:
            _root_span.set_inputs({"ticker": TICKER, "use_tuning": env["USE_TUNING"]})
        except Exception:
            pass
    _status, _code = _run_pipeline()
    if _root_span is not None:
        try:
            _root_span.set_outputs({"status": _status})
        except Exception:
            pass

sys.exit(_code)
