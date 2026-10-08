"""
model_sync.py — keep the served models current with the `models` branch.

Retraining runs in GitHub Actions (.github/workflows/daily-retrain.yml), which
force-pushes the registry, feature CSVs, MLflow store and the model's paper-trading
ledger (paper/) to the `models` branch.
On Render, ensure_fresh() downloads that branch's tarball at most every _TTL
seconds and unpacks it over the copies baked into the image. Local runs never
sync, so a local pipeline run isn't overwritten.
"""
import io
import os
import tarfile
import threading
import time
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent
_URL = os.environ.get(
    'MODELS_ARCHIVE_URL',
    'https://codeload.github.com/aarya127/quant_algorithms_ai/tar.gz/refs/heads/models')
_TTL = 6 * 3600
# Only these paths may be written from the archive
_ALLOWED = ('algorithms/machine_learning_algorithms/supervised/model_registry/',
            'algorithms/machine_learning_algorithms/data_pipelines/',
            'mlflow.db',
            'paper/')

_lock = threading.Lock()
_last_attempt = 0.0


def _on_render():
    # Same check as rate_limit.on_render, without its Flask import, so the
    # predictor also loads in the pipeline environment (paper model account).
    return bool(os.environ.get('RENDER') or os.environ.get('RENDER_EXTERNAL_URL')
                or os.environ.get('RENDER_SERVICE_ID'))


def ensure_fresh():
    global _last_attempt
    if not _on_render() or time.time() - _last_attempt < _TTL:
        return
    with _lock:
        if time.time() - _last_attempt < _TTL:
            return
        # Set before trying, so a failing download waits a full TTL to retry
        _last_attempt = time.time()
        try:
            resp = requests.get(_URL, timeout=30)
            resp.raise_for_status()
            n = 0
            with tarfile.open(fileobj=io.BytesIO(resp.content), mode='r:gz') as tar:
                for m in tar.getmembers():
                    rel = m.name.split('/', 1)[-1]   # drop the "<repo>-models/" prefix
                    if not m.isfile() or '..' in rel.split('/') or not rel.startswith(_ALLOWED):
                        continue
                    dest = _ROOT / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name + '.sync')
                    tmp.write_bytes(tar.extractfile(m).read())
                    os.replace(tmp, dest)   # readers never see a half-written file
                    n += 1
            print(f'[MODELS] synced {n} files from the models branch', flush=True)
        except Exception as e:
            print(f'[MODELS] sync failed, serving the current copy: {e}', flush=True)
