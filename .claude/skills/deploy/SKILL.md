---
name: deploy
description: Build, deploy, and troubleshoot quant_algorithms_ai on Docker/Render — the Gunicorn single-worker model, how the app pulls models from the `models` branch, secrets, health checks, and why a new route is inert until deployed. Use for any deploy question, Render config, or "why is my endpoint / GitHub Action failing" in this repo.
---

# Deployment (Docker + Render)

## How it's served

- **Image** (`Dockerfile`): `python:3.11-slim`, CPU-only PyTorch, FinBERT weights
  baked in at build time. Entry: `CMD ["./entrypoint.sh"]`.
- **`backend/entrypoint.sh`**:
  ```sh
  exec gunicorn app:app --bind "0.0.0.0:${PORT:-8080}" --workers "${GUNICORN_WORKERS:-1}" --worker-class sync --timeout 120
  ```
- **`render.yaml`** (Blueprint): Docker web service, `healthCheckPath: /health`.
  It still declares `plan: standard` and a 5 GB disk at `/app/mnt`, but the live
  service runs on the free tier (512 MB) and the disk is unused — nothing the app
  needs lives on it. Render auto-builds on every push and restarts on failure.

## Three facts that change how you write code

1. **Worker count** = `GUNICORN_WORKERS` (default 1; entrypoint honors it). The
   retrain job store is shared (SQLite, `backend/pipeline_store.py`), so >1 worker
   is correctness-safe now — but each sync worker can load FinBERT (~512 MB), so
   keep it at 1 on the free tier and raise only after upgrading RAM.
2. **A new Flask route is inert until committed AND deployed.** Editing
   `backend/app.py` in your working tree changes nothing on the live app. Render
   only picks up **pushed commits**.
3. **Served models come from the `models` branch**, not the image or a disk.
   Retraining runs in GitHub Actions and force-pushes that branch; on Render,
   `backend/model_sync.py` (called from `predictor.py` and `/api/mlflow/runs`)
   downloads its tarball at most every 6 h and unpacks only the model registry,
   `data_pipelines/` CSVs and `mlflow.db`. Local runs never sync. Look for
   `[MODELS]` lines in the Render logs.

## Secrets

Loaded in priority order: **env var → `keys.txt`** (gitignored). On Render, set
them in the dashboard (they're `sync: false` in `render.yaml`, so never committed):
`FINNHUB_API_KEY`, `POLYGON_API_KEY`, `FISCAL_AI_API_KEY`, `ALPACA_API_KEY`,
`ALPACA_SECRET_KEY`, `NVIDIA_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
Twitter keys, `CHART_IMG_KEY`, `ALERT_WEBHOOK_URL`.

Leave `PIPELINE_TRIGGER_TOKEN` unset on Render: `/api/pipeline/run` then refuses
with 503, which is intended (the free instance can't fit the pipeline).

The retrain workflow reads data-provider keys from repo secrets (Settings → Secrets
and variables → Actions): `FINNHUB_API_KEY`, `ALPHAVANTAGE_API_KEY`,
`POLYGON_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `NVIDIA_API_KEY`. A
missing key just skips that source. `RENDER_APP_URL` and `PIPELINE_TRIGGER_TOKEN`
repo secrets are no longer used.

## Deploy a change

```bash
git checkout -b <branch>       # never commit straight to main
git add -A && git commit -m "..."
git push -u origin <branch>    # open a PR; merging to main triggers Render build
```
Watch the Render dashboard build logs; `entrypoint.sh` used to run an import
pre-flight so import errors surface there. Confirm `GET /health` returns 200 after.

## Troubleshooting "Daily ML Pipeline: All jobs have failed"

The daily workflow (`daily-retrain.yml`) never touches Render — it runs the pipeline
on the Actions runner. The email means a step failed there; open the failing step's
log in the Actions tab and see the `retrain-pipeline` skill. Redeploying Render
won't fix it.

## Two Docker images

- **`Dockerfile`** — the web app (Flask + Gunicorn + PyTorch + FinBERT, ~3.5 GB).
  This is what Render builds.
- **`Dockerfile.pipeline`** — training-only (deps from `requirements-pipeline.txt`,
  no Flask/PyTorch/FinBERT, ~900 MB). `ENTRYPOINT` is `orchestrator.py`. Only the
  Kubernetes CronJob uses it; the GitHub Action installs `requirements-pipeline.txt`
  directly instead.

## Kubernetes (`k8s/`) — scaffolding, NOT the live deploy

A full manifest set (namespace `invest-ai`: deployment, service, ingress, hpa, pvc,
configmap, secret, cronjob) for an alternative K8s target. It uses **placeholder
image tags/hostnames** (`invest-ai:latest`, `example.com`), nothing in CI applies
it, and **Render is production**. Don't treat `k8s/` as authoritative. Note
`k8s/configmap.yaml` sets `GUNICORN_WORKERS=2`; if K8s ever goes live with >1 pod,
point `PIPELINE_DB_PATH` at the shared PVC so the SQLite job store is shared across
pods (a per-pod ephemeral DB would make `/status` 404 against the wrong pod).

## Three scheduling paths (only one is live)

| Path | Where | Status |
|---|---|---|
| `.github/workflows/daily-retrain.yml` | GitHub Actions runner → `models` branch | **Active — production** |
| `k8s/cronjob.yaml` (runs `Dockerfile.pipeline`) | Kubernetes | Unused scaffolding |
| `scripts/daily_predict.sh` | A local dev machine (hardcoded paths) | Local convenience |

Don't "fix" a failing retrain by touching the K8s cronjob or the local script — the
live one is the GitHub Action.

## Helper scripts (`scripts/`)

`build_cpp.sh` / `build_go.sh` / `build_all.sh` build the native components (under
`performance/`, not root — a past bug pointed them at nonexistent `cpp/`/`go/` dirs).
`daily_predict.sh` is a local-only cron helper with machine-specific paths.

## CI

`.github/workflows/ci.yml` runs `pytest` (light deps only) on every push/PR to
`main`. See the `testing` skill.

## Local run

```bash
cd backend && python app.py     # dev server on PORT (default 5001)
# or the production server:
PORT=8080 ./backend/entrypoint.sh
```

The polyglot build (`make all`, `make test`) covers C++/Go; see `ARCHITECTURE.md`.
