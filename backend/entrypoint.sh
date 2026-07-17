#!/bin/sh
# Workers default to 1. The retrain job store is now shared across processes
# (SQLite WAL, see pipeline_store.py), so >1 worker is safe for correctness — BUT
# each sync worker can lazily load FinBERT (~512 MB), so keep GUNICORN_WORKERS=1 on
# the 512 MB free tier and only raise it after upgrading the instance's RAM.
exec gunicorn app:app \
  --bind "0.0.0.0:${PORT:-8080}" \
  --workers "${GUNICORN_WORKERS:-1}" \
  --worker-class sync \
  --timeout 120
