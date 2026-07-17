---
name: retrain-pipeline
description: Run, trigger, or debug the ML retraining pipeline for a ticker in quant_algorithms_ai — the orchestrator, the /api/pipeline/* endpoints, the daily GitHub Actions retrain, and MLflow tracking. Use when asked to retrain a model, run the pipeline, understand the STEP/STATUS protocol, or debug the "Daily ML Pipeline failed" email.
---

# Retraining Pipeline

The retraining pipeline turns raw market data into promoted models. It runs the
same way locally, on demand via HTTP, and on a daily schedule.

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

## Output protocol (parser lives in backend/app.py)

The orchestrator prints a line protocol; `_run_retrain_job` in
`backend/routes/pipeline.py` parses it.
**If you change one side, change the other.**
```
STEP:<name>:start   STEP:<name>:done   LOG:<text>
STATUS:up_to_date   STATUS:done        STATUS:error:<name>
```
Exit 0 = done/up_to_date, exit 1 = a step failed.

## Trigger via the API (runs inside the Render web container)

```bash
curl -X POST "$RENDER_APP_URL/api/pipeline/run" \
  -H "Content-Type: application/json" -d '{"ticker":"NVDA"}'
# → {"job_id":"a1b2c3d4","status":"queued"}

curl "$RENDER_APP_URL/api/pipeline/status/a1b2c3d4"
# status: queued → running → done | up_to_date | error
```
The job runs in a background thread; state is persisted to a shared SQLite store
(`backend/pipeline_store.py`) — consistent across workers, bounded logs, old jobs
evicted. Trigger is single-flight (409 if one is running) and token-protected when
`PIPELINE_TRIGGER_TOKEN` is set. On the free tier the DB is ephemeral, so a full
restart still loses in-flight jobs (set `PIPELINE_DB_PATH` to a persistent disk).

## Daily schedule

`.github/workflows/daily-retrain.yml` — cron `0 2 * * 1-5` (02:00 UTC weekdays)
plus manual dispatch. It POSTs to `/api/pipeline/run` on the live app and polls
`/status` every 90 s. Requires the `RENDER_APP_URL` repo secret.

## Debugging "Daily ML Pipeline: All jobs have failed"

Failure in ~3 seconds = the first `curl` (trigger) step failed. In order of likelihood:

1. **Endpoints not deployed.** The `/api/pipeline/*` routes must be **committed and
   deployed to Render** — a working-tree-only edit does nothing. `curl -sf` gets a
   404 and exits non-zero. → commit + push + let Render redeploy.
2. **`RENDER_APP_URL` secret unset/wrong** → curl hits an empty/bad host, fails instantly.
3. **App down / failing health check** → check Render logs and `GET /health`.

A failure *minutes* in (not seconds) means the trigger worked but a pipeline step
errored — check the job's `last_logs` via `/api/pipeline/status/<job_id>` and the
Render logs for the `STATUS:error:<step>` line.

## Related

Promotion of the retrained model into serving is gated — see the `add-ml-model`
skill and `supervised/registry.py`.
