---
name: testing
description: Run and write tests for quant_algorithms_ai, and follow its conventions and polyglot (Python/C++/Go) layout. Use when asked to run the test suite, add a test, verify a change, or understand how the repo is structured and built.
---

# Testing & Conventions

## Running tests

| Command | What it runs |
|---|---|
| `pytest` (from repo root) | Unit/integration tests — `tests/test_*.py`. Fast, no network. **This is what CI runs.** |
| `cd tests && python3 run_all_tests.py` | Live-API smoke suite (Finnhub, AlphaVantage). **Needs API keys**, hits the network, rate-limited. Writes to `tests/results/`. |
| `make test` | Guarded C++ `ctest` + Go `go test` (skipped if not built) + `pytest`. |

Run `pytest` for logic changes. Reserve the API suite for verifying provider
integrations. See `tests/README.md` for the smoke-suite details.

> Historical gotcha: `make test` and `scripts/build_all.sh` used to point at
> nonexistent root `cpp/` and `go/` dirs (the code is under `performance/`), so
> they silently failed. Fixed — but if you see cwd errors from a Make target,
> check it references `performance/cpp_execution/` and `performance/go_services/`.

## CI (`.github/workflows/ci.yml`)

Runs on every push and PR to `main`. Installs **light deps only**
(numpy/pandas/sklearn/xgboost/lightgbm/pytest — no torch/transformers) and runs the
four pytest suites with coverage. So: **keep unit tests importable without heavy ML
deps** (that's why they mirror source logic instead of importing it). A separate
workflow, `daily-retrain.yml`, handles scheduled retraining (see `retrain-pipeline`).

## The unit-test convention (important)

Tests **mirror source logic as pure, self-contained functions** — they do not
import heavy modules, hit the network, or read real data files. Each re-implements
or isolates the behaviour under test against synthetic data. Examples:

- `test_transforms.py` — mirrors `clean.py` logic
- `test_targets.py` — mirrors `normalize.py` target construction; asserts no look-ahead
- `test_evaluation_gate.py` — duplicates the registry promotion-gate constants as a pure `evaluate_gate`
- `test_pipeline_integration.py` — full clean→normalize on a synthetic 80-row frame, no network

`tests/conftest.py` adds `backend/`, `supervised/`, and `data_pipelines/` to
`sys.path` so those modules import directly.

**New tests:** add `tests/test_<area>.py` as pytest files following the mirror
pattern. If you change a contract that a test mirrors (e.g. the target definitions
in `normalize.py`, or the promotion-gate thresholds in `registry.py`), update the
corresponding test in the same change.

## Verifying a real change

Beyond tests, drive the actual flow when it matters:
- Pipeline change → run `orchestrator.py NVDA` once (see `retrain-pipeline` skill).
- Backend route → run `cd backend && python app.py` and curl the endpoint.
- Remember a route only affects production once deployed (see `deploy` skill).

## Code conventions

- Match the surrounding file's style, naming, and comment density.
- Backend routes: wrap the body in try/except returning `jsonify({'error': ...}), 500`.
- Pipeline scripts: `python <script>.py TICKER` (default `NVDA`), paths via
  `Path(__file__).parent`, progress as plain stdout lines, hand off via `<SYM>_*.csv`.
- Never hardcode secrets — env var → `keys.txt` (gitignored).

## Polyglot layout & build

`ARCHITECTURE.md` is the reference. Roughly: Python ~70% (strategy, research,
serving), C++ for ultra-low-latency execution (`performance/cpp_execution/`,
pybind11), Go for concurrent risk services (`performance/go_services/`, gRPC).

| Make target | Action |
|---|---|
| `make python` | `pip install -r requirements.txt` |
| `make cpp` | build C++ engine (`scripts/build_cpp.sh`) |
| `make go` | build Go services (`scripts/build_go.sh`) |
| `make all` | all three |
| `make run-backend` / `make dev` | start Flask (`backend/app.py`) |
| `make run-risk` | start the Go risk engine |
| `make clean` / `make rebuild` | remove / rebuild artefacts |
