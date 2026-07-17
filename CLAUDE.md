# CLAUDE.md

This project's coding-agent guide is **[AGENTS.md](AGENTS.md)** — read it first.
It's the single source of truth for repo conventions, the ML pipeline contract,
deployment facts, and gotchas. This file only adds Claude Code specifics.

## Skills

Task-specific playbooks are available as skills in [`.claude/skills/`](.claude/skills/)
and load automatically when relevant:

- **retrain-pipeline** — run the ML retraining pipeline; the `/api/pipeline/*`
  endpoints; the daily GitHub Actions retrain; MLflow.
- **add-ml-model** — add a supervised model or a new pipeline stage so it plugs
  into the existing feature-matrix → registry flow.
- **deploy** — Docker/Render build, the single-worker Gunicorn model, secrets,
  and diagnosing the "Daily ML Pipeline failed" email.
- **testing** — run the test suites and follow the pure-function test convention.

## Fast facts (details in AGENTS.md)

- A new Flask route is **inert until committed + deployed to Render**. That's the
  usual root cause of the daily-retrain failure email — deploy, don't just edit.
- **Workers** = `GUNICORN_WORKERS` (default 1). The retrain job store is shared
  (SQLite, `backend/pipeline_store.py`), so >1 worker is correctness-safe — but keep
  it at 1 on the free tier (each worker can load FinBERT ~512 MB).
- Pipeline stages hand off via `<SYMBOL>_*.csv` files; the 6 canonical targets are
  defined once in `data_pipelines/normalize.py`.
- Trained models live in `supervised/model_registry/`, not top-level `models/`.
- Tests mirror source logic as pure, network-free functions (`tests/test_*.py`).

## Etiquette

- Don't commit or push unless asked; branch off `main` first if you do.
- Never write secrets into source — env vars → `keys.txt` (gitignored) only.
