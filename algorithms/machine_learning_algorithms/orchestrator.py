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


def _run_step(name, cmd):
    """
    Run a pipeline step as a subprocess.

    Returns (returncode, up_to_date).
    Streams LOG: lines for each output line.
    """
    print(f"STEP:{name}:start", flush=True)
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
    return proc.returncode, up_to_date


for step_name, step_cmd in STEPS:
    rc, up_to_date = _run_step(step_name, step_cmd)

    if rc != 0:
        print(f"STATUS:error:{step_name}", flush=True)
        sys.exit(1)

    print(f"STEP:{step_name}:done", flush=True)

    if up_to_date:
        # Feature data is already current for today.
        # Skip retraining — the existing registry is still valid.
        print("STATUS:up_to_date", flush=True)
        sys.exit(0)

print("STATUS:done", flush=True)
