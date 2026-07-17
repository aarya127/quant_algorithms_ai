# AGENTS.md — Guide for Coding Agents

Conventions, contracts, and gotchas for AI coding agents (Claude Code, GitHub
Copilot, Cursor) and human contributors working in `quant_algorithms_ai`.
This is the vendor-neutral source of truth; `CLAUDE.md` points here.

Start with the [README](README.md) for what the system *is*. This file is about
how to *change* it safely. Task-specific playbooks live in
[`.claude/skills/`](.claude/skills/).

---

## The one thing that bites everyone first

**Adding a Flask route in `backend/app.py` does nothing in production until it is
committed AND deployed to Render.** The daily retraining workflow calls
`/api/pipeline/run` on the *live* Render app. If those endpoints only exist in
your working tree (or an uncommitted diff), the workflow fails in ~3 seconds on
the first `curl`, and you get a "Daily ML Pipeline: All jobs have failed" email.
The fix for that class of failure is almost always **commit + push + let Render
redeploy**, not a code change. Also confirm the `RENDER_APP_URL` repo secret is set.

---

## Repo map (where things live)

| Area | Path | Notes |
|---|---|---|
| Flask backend | `backend/app.py` + `backend/routes/` | 31 routes; pipeline/charts/news live in blueprints, rest still inline in app.py |
| Prediction serving + drift | `backend/predictor.py` | importable functions; **HTTP routes not yet wired** |
| Retraining driver | `algorithms/machine_learning_algorithms/orchestrator.py` | 5-step pipeline |
| Pipeline stages | `.../data_pipelines/` | `run_pipeline.py`, `clean.py`, `normalize.py` |
| Supervised models | `.../supervised/` | `models.py`, `registry.py`, `main.py` |
| Trained artefacts | `.../supervised/model_registry/<TICKER>/` | **not** in top-level `models/` |
| Unsupervised regimes | `.../unsupervised/unsupervised.py` | single script |
| Feature allow-list | `.../factor_discovery/output/recommended_features.txt` | consumed by supervised + unsupervised |
| Volatility stack | `algorithms/volatility_forecasting/` | SABR/Heston, signals, backtest, portfolio |
| LLM layer | `ai_platform/` | router, narrator, judge, nvidia |
| Data providers | `data/` | one module per API — ref: [docs/data-providers.md](docs/data-providers.md) |
| Sentiment | `sentiment/` | FinBERT US + Canadian |
| Time-series research | `.../time_series_models/`, `.../eda/` | standalone scripts — ref: [docs/research-scripts.md](docs/research-scripts.md) |
| High-perf | `performance/cpp_execution/`, `performance/go_services/` | C++ (pybind11), Go (gRPC) — ref: [docs/performance.md](docs/performance.md) |
| Tests | `tests/` | pytest + API smoke tests |
| Deploy | `Dockerfile`, `render.yaml`, `backend/entrypoint.sh` | Render + Docker |
| CI / schedule | `.github/workflows/{ci.yml,daily-retrain.yml}` | |
| MLflow store | `mlflow.db`, `mlruns/` | SQLite, repo root |

`models/` (top-level) is pricing-**theory** scaffolding, mostly placeholders. Do
not put trained ML artefacts there — they belong in the supervised model registry.

---

## The pipeline contract (read before touching ML code)

Everything is keyed off two conventions. Break either and the pipeline silently
misbehaves.

1. **CSV filename prefix `<SYMBOL>_`.** Stages hand off via CSVs in
   `data_pipelines/`, in this order:
   ```
   <SYM>_features.csv → _features_clean.csv → _features_normalized.csv → _features_with_regimes.csv
   ```
   Plus `<SYM>_scaler.pkl` and `<SYM>_targets.csv` from the normalize step.

2. **The 6 canonical targets**, defined once in `data_pipelines/normalize.py`
   (single source of truth; mirrored in `tests/test_targets.py`):
   - Regression: `target_1d`, `target_5d`, `target_vol_5d`
   - Classification: `target_dir_1d`, `target_large_move`, `target_regime`

Every stage is a standalone script invoked as `python <script>.py TICKER`,
defaulting the ticker to `NVDA`, resolving paths via `Path(__file__).parent` so
it runs from any cwd. New stages must follow this.

**Orchestrator output protocol** (parsed by `app.py`; keep in sync if you change either side):
```
STEP:<name>:start | STEP:<name>:done | LOG:<text>
STATUS:up_to_date | STATUS:done | STATUS:error:<name>
```
Exit code 0 = done/up_to_date, 1 = a step failed.

---

## Common tasks (short version — full playbooks in `.claude/skills/`)

- **Add a supervised model** → extend the factory dicts in
  `supervised/models.py` (`reg_model_set()` / `clf_model_set()`). It flows
  automatically through walk-forward → holdout → MLflow → registry. To become
  servable it must be best for a registered target *and* clear the promotion gate.
- **Add a pipeline stage** → new `data_pipelines/<stage>.py TICKER` reading the
  previous `<SYM>_*.csv` and writing the next, then register it in
  `orchestrator.py`'s `STEPS` list. Emit plain lines (the orchestrator wraps them as `LOG:`).
- **Add a research quant algo** → new dir `algorithms/<family>/<concept>/` with
  `theory.tex` + `prototype.py` (or a sub-package with `schemas.py` + `run_*.py`
  like `volatility_forecasting`).
- **Add an API route** → `backend/app.py`, wrap the body in try/except returning
  `jsonify({'error': ...}), 500`, and remember it's inert until deployed.

---

## Deployment facts that change how you code

- **Worker count** is `GUNICORN_WORKERS` (default 1; `backend/entrypoint.sh` honors it).
  The retrain job store is now shared (SQLite, `backend/pipeline_store.py`), so >1
  worker is safe for correctness — but each sync worker can load FinBERT (~512 MB),
  so keep it at 1 on small instances and raise only after upgrading RAM.
- **Persistent disk** is mounted at `/app/mnt` on Render and holds models +
  feature CSVs. It's why retraining runs *inside* the web container (a separate
  Render cron job couldn't share the disk).
- **Port**: local dev defaults to `5001`; Docker/Render use `8080` via `PORT`.
- FinBERT weights are baked into the image at build time — don't add runtime downloads.
- **Two Docker images**: `Dockerfile` (web app, what Render builds) and
  `Dockerfile.pipeline` (training-only, used only by the K8s CronJob).
- **Three scheduling paths, one live**: `.github/workflows/daily-retrain.yml`
  (Render, **production**) vs `k8s/cronjob.yaml` (unused scaffolding) vs
  `scripts/daily_predict.sh` (local machine only). Fix a failing retrain via the
  GitHub Action, not the other two.
- **`k8s/` is scaffolding** — placeholder image tags/hostnames, applied by nothing.
  Render (`render.yaml`) is the real deploy. Don't cite k8s manifests as truth.
- **CI** = `.github/workflows/ci.yml`: `pytest` with light deps only (no
  torch/transformers) on every push/PR. Keep unit tests importable without heavy deps.

## Component status (don't trust the tree at face value)

The repo mixes shipped code, standalone research, and empty scaffolding. Before
"fixing" or extending something, check it's real:

- **Wired & working**: `backend/`, `data/`, `sentiment/`, `ai_platform/` (except
  `signal_narrator.py`), the ML pipeline + registry + MLflow, `volatility_forecasting/`.
- **Real but standalone** (not called by app/pipeline): `time_series_models/`,
  `eda/`, `factor_discovery/`, `greeks/`, `macd_rsi/`, and `performance/`
  (C++/Go — real code, but not imported by the deployed app; their READMEs
  overstate the dir layout).
- **Not yet wired**: `backend/predictor.py` (functions exist, no routes),
  `ai_platform/signal_narrator.py` (no caller).
- **Empty placeholders** (imply features that don't exist): all of top-level
  `models/`, `deep_learning/`, `monte_carlo/`, `data/streaming/`, `backtesting/*.py`,
  and several 0-byte `prototype.py` stubs. Don't document these as working.

---

## Testing & conventions

- **Run tests:** `pytest` for unit/integration (`tests/test_*.py`);
  `cd tests && python3 run_all_tests.py` for the live-API smoke suite (needs keys);
  `make test` runs C++ `ctest` + Go `go test` + the API suite.
- **Unit-test convention:** tests **mirror source logic as pure, self-contained
  functions** — no network, no disk, no external data. See `test_transforms.py`,
  `test_targets.py`, `test_evaluation_gate.py`, `test_pipeline_integration.py`.
  `conftest.py` puts `backend/`, `supervised/`, and `data_pipelines/` on `sys.path`.
- New unit tests go in `tests/test_<area>.py`.
- **Style:** match the surrounding file. Backend routes use try/except → JSON
  error. Pipeline scripts print progress as plain stdout lines. Paths are always
  resolved relative to `__file__`, never hardcoded or cwd-relative.

## Polyglot layout

Python is ~70% (strategy/research/serving). C++ (`performance/cpp_execution/`,
pybind11) is for ultra-low-latency execution; Go (`performance/go_services/`,
gRPC) for concurrent risk services. `ARCHITECTURE.md` covers the split and the
Python↔C++↔Go integration points. Build via `make {cpp,go,python,all}`.

---

## Safety / do-not

- **Never commit secrets.** Keys load from env vars → `keys.txt` (gitignored).
  No secret literals in source, ever.
- **Don't commit or push unless asked.** If asked, branch off `main` first.
- **Don't retrain/overwrite the model registry casually** — the promotion gate
  and `prev/` snapshots exist to prevent bad models reaching production. Use
  `force=True` only deliberately.
- **Don't edit `mlflow.db` or files under `mlruns/` by hand.**
- Large data CSVs, `.pkl` artefacts, and `mlflow.db` may be tracked — check
  `.gitignore` before adding generated files.
