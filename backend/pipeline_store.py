"""
pipeline_store.py — durable, bounded store for retrain-job state.

Replaces the old in-process `_PIPELINE_JOBS` dict, which (a) grew unbounded,
(b) was wiped on every worker restart, and (c) was correct only under a single
gunicorn worker. This is a small SQLite store in WAL mode, so:

  * job status survives a worker recycle (as long as the DB file survives — on
    Render's free tier the filesystem is ephemeral, so point PIPELINE_DB_PATH at
    the persistent disk once you're on a paid plan for true cross-restart durability);
  * POST /run and GET /status stay consistent across multiple gunicorn workers;
  * per-job logs are capped and old jobs are evicted, so memory/disk stay bounded.

All functions open a short-lived connection, so they're safe to call from the
request threads and the background retrain thread(s) concurrently.
"""
import os
import json
import sqlite3
import datetime
import threading

_DB_PATH = os.environ.get(
    "PIPELINE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "pipeline_state.db"),
)
_LOG_CAP = 200     # keep only the most recent N log lines per job
_JOB_CAP = 50      # keep only the most recent N jobs overall
_ACTIVE = ("queued", "running")

# Serialises append_log's read-modify-write within this process; WAL + busy
# timeout handle concurrency across processes.
_LOG_LOCK = threading.Lock()


def _connect():
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the table if needed. Safe to call at import / startup repeatedly."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id       TEXT PRIMARY KEY,
                ticker       TEXT,
                status       TEXT,
                current_step TEXT,
                error        TEXT,
                timestamp    TEXT,
                logs         TEXT,
                updated_at   TEXT
            )
            """
        )


def _now():
    return datetime.datetime.now().isoformat()


def create_job(job_id, ticker):
    """Insert a new queued job and evict old finished jobs."""
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO jobs "
            "(job_id, ticker, status, current_step, error, timestamp, logs, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (job_id, ticker, "queued", "queued", None, now, "[]", now),
        )
        # Evict all but the most recent _JOB_CAP jobs.
        conn.execute(
            "DELETE FROM jobs WHERE job_id NOT IN "
            "(SELECT job_id FROM jobs ORDER BY updated_at DESC LIMIT ?)",
            (_JOB_CAP,),
        )


def active_job():
    """Return the first queued/running job as a dict, or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status IN (?, ?) ORDER BY updated_at DESC LIMIT 1",
            _ACTIVE,
        ).fetchone()
    return _row_to_dict(row) if row else None


def set_status(job_id, status=None, current_step=None, error=None):
    """Update whichever fields are provided (non-None)."""
    sets, vals = [], []
    if status is not None:
        sets.append("status = ?"); vals.append(status)
    if current_step is not None:
        sets.append("current_step = ?"); vals.append(current_step)
    if error is not None:
        sets.append("error = ?"); vals.append(error)
    if not sets:
        return
    sets.append("updated_at = ?"); vals.append(_now())
    vals.append(job_id)
    with _connect() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", vals)


def append_log(job_id, line):
    """Append a log line, keeping only the most recent _LOG_CAP lines."""
    with _LOG_LOCK, _connect() as conn:
        row = conn.execute("SELECT logs FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return
        logs = json.loads(row["logs"] or "[]")
        logs.append(line)
        if len(logs) > _LOG_CAP:
            logs = logs[-_LOG_CAP:]
        conn.execute(
            "UPDATE jobs SET logs = ?, updated_at = ? WHERE job_id = ?",
            (json.dumps(logs), _now(), job_id),
        )


def get_job(job_id):
    """Return the job as a dict (with `logs` as a list), or None."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def _row_to_dict(row):
    d = dict(row)
    d["logs"] = json.loads(d.get("logs") or "[]")
    return d
