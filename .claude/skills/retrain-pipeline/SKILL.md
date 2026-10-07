---
name: retrain-pipeline
description: Run, trigger, or debug the ML retraining pipeline for a ticker in quant_algorithms_ai — the orchestrator, the /api/pipeline/* endpoints, the daily GitHub Actions retrain, and MLflow tracking. Use when asked to retrain a model, run the pipeline, understand the STEP/STATUS protocol, or debug the "Daily ML Pipeline failed" email.
---

# Retraining Pipeline

The retraining pipeline turns raw market data into promoted models. The same
orchestrator runs locally (CLI, or the ML Signals tab's "Run Pipeline" button via
HTTP) and on a daily schedule in GitHub Actions — never inside the Render app.

## The 5 steps

`algorithms/machine_learning_algorithms/orchestrator.py` chains these as
subprocesses. Each reads the previous stage's CSV and writes the next, keyed by a
`<SYMBOL>_` prefix in `data_pipelines/`:

| # | Step | Script | Writes |
|---|---|---|---|
| 1 | extract | `data_pipelines/run_pipeline.py TICKER [PERIOD]` | `<SYM>_features.csv` (incremental append; `PERIOD=full` forces rebuild) |
| 2 | clean | `data_pipelines/clean.py TICKER` | `<SYM>_features_clean.csv` |
| 3 | normalize | `data_pipelines/normalize.py TICKER` | `<SYM>_features_normalized.csv`, `<SYM>_scaler.pkl`, `<SYM>_targets.csv` |
| 4 | unsupervised | `unsupervised/unsupervised.py TICKER` | `<SYM>_features_with_regimes.csv` |
| 5 | supervised | `supervised/supervised.py TICKER` | model registry + `supervised/output/` |

After a successful supervised step the orchestrator runs the LLM-as-judge
(`ai_platform/llm_judge.py`).

## Run it locally

```bash
python algorithms/machine_learning_algorithms/orchestrator.py NVDA
# Hyperparameter tuning (slower, better): prepend USE_TUNING=1
USE_TUNING=1 python algorithms/machine_learning_algorithms/orchestrator.py NVDA
```
Ticker defaults to `NVDA`. The run is wrapped in an MLflow trace (one child span
per step). Browse it: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Output protocol (parser lives in backend/routes/pipeline.py)

The orchestrator prints a line protocol; `_run_retrain_job` in
`backend/routes/pipeline.py` parses it.
**If you change one side, change the other.**
```
STEP:<name>:start   STEP:<name>:done   LOG:<text>
STATUS:up_to_date   STATUS:done        STATUS:error:<name>
```
Exit 0 = done/up_to_date, exit 1 = a step failed.

## Trigger via the API (local runs)

```bash
curl -X POST "http://localhost:5001/api/pipeline/run" \
  -H "Content-Type: application/json" -d '{"ticker":"NVDA"}'
# → {"job_id":"a1b2c3d4","status":"queued"}

curl "http://localhost:5001/api/pipeline/status/a1b2c3d4"
# status: queued → running → done | up_to_date | error
```
The job runs in a background thread; state is persisted to a shared SQLite store
(`backend/pipeline_store.py`) — consistent across workers, bounded logs, old jobs
evicted. Trigger is single-flight (409 if one is running). On Render it refuses
with 503 while `PIPELINE_TRIGGER_TOKEN` is unset — the intended production state;
with a token set, callers must send it as the `X-Pipeline-Token` header.

## Daily schedule

`.github/workflows/daily-retrain.yml` — cron `0 2 * * 1-5` (02:00 UTC weekdays)
plus manual `workflow_dispatch` with a `ticker` input (default `NVDA`). It runs
`orchestrator.py` on the Actions runner (the free 512 MB Render instance OOMed
running it in-process):

1. Restore the model registry and `mlflow.db` from the `models` branch. Everything
   else is a full rebuild — a fresh checkout has no feature CSVs to append to.
2. Run the pipeline.
3. Force-push a single-commit `models` branch holding
   `supervised/model_registry/**`, `data_pipelines/<TICKER>_features_with_regimes.csv`
   and `mlflow.db`.

The web app pulls that branch on Render at most every 6 h (`backend/model_sync.py`);
local runs never sync. Data-provider keys come from repo secrets (`FINNHUB_API_KEY`,
`ALPHAVANTAGE_API_KEY`, `POLYGON_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`,
`NVIDIA_API_KEY`); a missing one just skips that source.

## Debugging "Daily ML Pipeline: All jobs have failed"

The pipeline itself failed on the Actions runner — Render isn't involved, so
redeploying won't help. Open the run in the Actions tab and read the failing step's
log: "Run pipeline" failures end with a `STATUS:error:<step>` line naming the stage;
reproduce locally with `orchestrator.py <TICKER>`.

## Related

Promotion of the retrained model into serving is gated — see the `add-ml-model`
skill and `supervised/registry.py`.
