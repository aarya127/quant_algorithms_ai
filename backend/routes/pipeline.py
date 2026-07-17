"""
routes/pipeline.py — Pipeline Job Management blueprint.

POST /api/pipeline/run and GET /api/pipeline/status/<job_id>, used by the
scheduled-retraining GitHub Actions workflow (.github/workflows/daily-retrain.yml).
Extracted from app.py; behavior unchanged.
"""
import os
import sys
import time
import uuid
import threading
import subprocess
from pathlib import Path

from flask import Blueprint, jsonify, request

# Retrain-job state lives in a small SQLite store (see pipeline_store.py): durable
# across worker recycles, consistent across gunicorn workers, with bounded logs and
# automatic eviction of old jobs. Point PIPELINE_DB_PATH at the persistent disk for
# true cross-restart durability on paid Render plans.
import pipeline_store

bp = Blueprint('pipeline', __name__)

pipeline_store.init_db()

# quant_algorithms_ai/ — this file lives at backend/routes/pipeline.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Shared secret protecting the (expensive) retrain trigger. When set, callers must
# send a matching `X-Pipeline-Token` header; the GitHub Actions workflow sends it
# from the PIPELINE_TRIGGER_TOKEN repo secret. When unset, the endpoint stays open
# but logs a warning — set it in the Render dashboard to lock down this public app.
_PIPELINE_TOKEN = os.environ.get('PIPELINE_TRIGGER_TOKEN', '').strip()

# Hard ceiling on a single retrain run (seconds). Overridable via env; prevents a
# hung orchestrator step from pinning a worker/subprocess indefinitely.
_RETRAIN_TIMEOUT = int(os.environ.get('PIPELINE_TIMEOUT_SECONDS', str(3 * 60 * 60)))


def _run_retrain_job(job_id, ticker):
    """Run ML retraining as a subprocess, parsing orchestrator output protocol."""
    proc = None
    try:
        pipeline_store.set_status(job_id, status='running', current_step='initializing')
        print(f"[PIPELINE] Starting retraining for {ticker} (job {job_id})", flush=True)

        orchestrator_script = str(
            _PROJECT_ROOT / 'algorithms' / 'machine_learning_algorithms' / 'orchestrator.py'
        )
        cmd = [sys.executable, orchestrator_script, ticker]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(_PROJECT_ROOT),
        )

        deadline = time.monotonic() + _RETRAIN_TIMEOUT
        up_to_date = False

        # Parse orchestrator output protocol:
        #   STEP:<name>:start|done · LOG:<text> · STATUS:up_to_date|done|error:<name>
        for line in proc.stdout:
            if time.monotonic() > deadline:
                raise TimeoutError(f"retrain exceeded {_RETRAIN_TIMEOUT}s")
            line = line.rstrip()

            if line.startswith('STEP:'):
                parts = line.split(':')
                if len(parts) >= 3:
                    step_name, step_status = parts[1], parts[2]
                    if step_status == 'start':
                        pipeline_store.set_status(job_id, current_step=step_name)
                        pipeline_store.append_log(job_id, f"Starting step: {step_name}")
                    elif step_status == 'done':
                        pipeline_store.append_log(job_id, f"Completed step: {step_name}")

            elif line.startswith('STATUS:'):
                status_msg = line.split(':', 1)[1]
                if status_msg == 'up_to_date':
                    up_to_date = True
                    pipeline_store.append_log(job_id, "Data already up to date")
                elif status_msg == 'done':
                    pipeline_store.append_log(job_id, "Pipeline completed successfully")
                elif status_msg.startswith('error:'):
                    error_step = status_msg.split(':', 1)[1]
                    pipeline_store.set_status(job_id, error=f"Step '{error_step}' failed")
                    pipeline_store.append_log(job_id, f"Error in step: {error_step}")

            elif line.startswith('LOG:'):
                pipeline_store.append_log(job_id, line[4:])

            elif line.strip():
                pipeline_store.append_log(job_id, line)

        proc.wait(timeout=60)

        if proc.returncode == 0:
            final_status = 'up_to_date' if up_to_date else 'done'
            job = pipeline_store.get_job(job_id)
            step = 'completed' if not (job and job.get('error')) else None
            pipeline_store.set_status(job_id, status=final_status, current_step=step)
        else:
            pipeline_store.set_status(job_id, status='error')

        print(f"[PIPELINE] {'✓' if proc.returncode == 0 else '✗'} Retraining "
              f"{'completed' if proc.returncode == 0 else 'failed'} for {ticker} (job {job_id})",
              flush=True)

    except Exception as e:
        print(f"[PIPELINE] ✗ Exception retraining {ticker} (job {job_id}): {e}", flush=True)
        import traceback
        traceback.print_exc()
        pipeline_store.set_status(job_id, status='error', error=str(e), current_step='failed')
        pipeline_store.append_log(job_id, f"Exception: {str(e)}")
    finally:
        # Never orphan the orchestrator (and its child step subprocesses).
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:
                pass


@bp.route('/api/pipeline/run', methods=['POST'])
def pipeline_run():
    """Start an ML retraining job for a given ticker."""
    try:
        # Auth: retraining is expensive, so protect it behind a shared secret.
        if _PIPELINE_TOKEN:
            if request.headers.get('X-Pipeline-Token', '') != _PIPELINE_TOKEN:
                return jsonify({'success': False, 'error': 'unauthorized'}), 401
        else:
            print("[PIPELINE] WARNING: /api/pipeline/run is unprotected "
                  "(PIPELINE_TRIGGER_TOKEN not set).", flush=True)

        data = request.get_json(force=True) or {}
        ticker = data.get('ticker', 'NVDA').upper().strip()

        if not ticker:
            return jsonify({'success': False, 'error': 'ticker is required'}), 400

        # Single-flight: refuse to start a second retrain while one is active, so a
        # burst of triggers can't spawn parallel pipelines and OOM the instance.
        active = pipeline_store.active_job()
        if active is not None:
            return jsonify({
                'success': False,
                'error': 'a retrain job is already in progress',
                'job_id': active['job_id'],
                'status': active['status'],
            }), 409

        # Generate job ID and persist it
        job_id = str(uuid.uuid4())[:8]
        pipeline_store.create_job(job_id, ticker)

        # Start background thread
        thread = threading.Thread(target=_run_retrain_job, args=(job_id, ticker), daemon=True)
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'ticker': ticker,
            'status': 'queued'
        })

    except Exception as e:
        print(f"[PIPELINE] Error in /api/pipeline/run: {e}", flush=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/pipeline/status/<job_id>')
def pipeline_status(job_id):
    """Check the status of a pipeline job."""
    try:
        job = pipeline_store.get_job(job_id)
        if job is None:
            return jsonify({'success': False, 'error': f'Job {job_id} not found'}), 404

        response = {
            'success': True,
            'job_id': job_id,
            'ticker': job['ticker'],
            'status': job['status'],
            'current_step': job['current_step'],
            'timestamp': job['timestamp'],
        }
        if job.get('error'):
            response['error'] = job['error']
        if job.get('logs'):
            response['last_logs'] = job['logs'][-5:]

        return jsonify(response)

    except Exception as e:
        print(f"[PIPELINE] Error in /api/pipeline/status: {e}", flush=True)
        return jsonify({'success': False, 'error': str(e)}), 500
