"""
routes/ml.py — ML Signals tab: the model registry, served through predictor.py.

GET /api/predict/<ticker>, /api/drift/<ticker>, /api/model/status and
/api/mlflow/runs/<ticker>. Responses are 200 with success:false when a ticker has
no trained models yet, so the panels can show the reason.
"""
import re
from pathlib import Path

from flask import Blueprint, jsonify

bp = Blueprint('ml', __name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ticker(raw):
    """Upper-cased ticker, or None — it becomes part of file paths."""
    t = raw.upper()
    return t if re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,9}', t) else None


def _no_models(t):
    # Feature CSVs are gitignored, so they exist only where the pipeline has run
    return jsonify({'success': False,
                    'error': f'No pipeline data for {t} on this server yet — run the pipeline first.'})


@bp.route('/api/predict/<ticker>')
def predict(ticker):
    t = _ticker(ticker)
    if not t:
        return jsonify({'success': False, 'error': 'invalid ticker'}), 400
    import predictor
    try:
        return jsonify({'success': True, **predictor.predict_latest(t)})
    except FileNotFoundError:
        return _no_models(t)


@bp.route('/api/drift/<ticker>')
def drift(ticker):
    t = _ticker(ticker)
    if not t:
        return jsonify({'success': False, 'error': 'invalid ticker'}), 400
    import predictor
    try:
        report = predictor.check_drift(t)
    except FileNotFoundError:
        return _no_models(t)
    return jsonify({'success': 'error' not in report, **report})


@bp.route('/api/model/status')
def model_status():
    import predictor
    status = predictor.model_status()
    if 'error' in status:
        return jsonify({'success': False, 'error': status['error']})
    return jsonify({'success': True, 'registry': status})


@bp.route('/api/mlflow/runs/<ticker>')
def mlflow_runs(ticker):
    t = _ticker(ticker)
    if not t:
        return jsonify({'success': False, 'error': 'invalid ticker'}), 400
    # Opening a missing SQLite store would create an empty one; report instead.
    if not (_PROJECT_ROOT / 'mlflow.db').exists():
        return jsonify({'success': True, 'runs': [],
                        'message': 'No MLflow runs recorded on this server yet.'})
    import predictor  # noqa: F401  (puts supervised/ on sys.path)
    from mlflow.tracking import MlflowClient
    from mlflow_tracker import TRACKING_URI

    client = MlflowClient(tracking_uri=TRACKING_URI)
    exp = client.get_experiment_by_name(t)
    runs = [] if exp is None else client.search_runs(
        [exp.experiment_id], order_by=['attributes.start_time DESC'], max_results=50)
    return jsonify({'success': True, 'runs': [{
        'run_name': r.info.run_name or '',
        'tags':     {k: v for k, v in r.data.tags.items() if not k.startswith('mlflow.')},
        'metrics':  r.data.metrics,
        'params':   r.data.params,
    } for r in runs]})
