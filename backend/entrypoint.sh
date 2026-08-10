#!/bin/sh
# Workers default to 1. The retrain job store is now shared across processes
# (SQLite WAL, see pipeline_store.py), so >1 worker is safe for correctness — BUT
# each worker process can lazily load FinBERT (~512 MB), so keep GUNICORN_WORKERS=1 on
# the 512 MB free tier and only raise it after upgrading the instance's RAM.
# Concurrency instead comes from gthread threads: they share one process (and one
# FinBERT), so parallel API calls from the frontend no longer serialize behind a
# single sync worker.
exec gunicorn app:app \
  --bind "0.0.0.0:${PORT:-8080}" \
  --workers "${GUNICORN_WORKERS:-1}" \
  --worker-class gthread \
  --threads "${GUNICORN_THREADS:-8}" \
  --timeout 120
