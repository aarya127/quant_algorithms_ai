---
name: add-ml-model
description: Add a new supervised ML model, a new pipeline stage, or a new research quant algorithm to quant_algorithms_ai so it plugs into the existing feature-matrix → training → registry flow. Use when asked to add/extend a model, add a pipeline step, or scaffold a new algorithm, and to understand the model registry promotion gate.
---

# Add a Model, Stage, or Algorithm

Follow the existing contracts — the pipeline is convention-driven, so plugging in
correctly means it "just works" end to end.

## Add a supervised model (most common)

Models are declared as factory "ladders" in
`algorithms/machine_learning_algorithms/supervised/models.py`:

- `reg_model_set()` — regression models (ridge, elasticnet, huber, rf, xgb, lgb)
- `clf_model_set()` — classification models (logistic, svc_cal, rf, xgb, lgb)

**To add one:** add an entry to the relevant factory dict. Everything downstream
is automatic — `pipeline.py` runs it through walk-forward CV
(`data.py: make_walk_forward_splits`) and holdout, `mlflow_tracker.py` logs it,
and the best model per target is offered to the registry.

Keep it inside the sklearn-style `fit`/`predict` (or `predict_proba`) interface
the ladder expects; don't special-case it in `pipeline.py`.

### Targets are fixed

The 6 canonical targets are defined once in `data_pipelines/normalize.py`:
- Regression: `target_1d`, `target_5d`, `target_vol_5d`
- Classification: `target_dir_1d`, `target_large_move`, `target_regime`

Registered/servable targets (`_SAVE_TARGETS` in `registry.py`): `target_1d`,
`target_5d`, `target_vol_5d`, `target_dir_1d`, `target_regime`. Don't invent a new
target without updating `normalize.py` (and `tests/test_targets.py`, which mirrors it).

### Getting into serving: the promotion gate

A trained model only replaces production if it clears **all three** checks in
`registry.py._promotion_gate` (unless `force=True`):
1. Absolute floor — regression `ic > 0.02`, classification `f1_w > 0.30`
2. Beats the naive baseline by ≥ 5% (`_BASELINE_BEAT_RATIO = 1.05`)
3. Improves over current production by ≥ 0.005 (`_MIN_DELTA`)

The outgoing model is snapshotted to `prev/` first (enables `rollback_registry()`).
Artefacts land in `supervised/model_registry/<TICKER>/<target>/`:
`model.pkl`, scalers, `features.json`, `train_stats.json`, `metadata.json`.

## Add a pipeline stage

1. Create `data_pipelines/<stage>.py` runnable as `python <stage>.py TICKER`
   (default `NVDA`, paths via `Path(__file__).parent`).
2. Read the previous stage's `<SYM>_*.csv`, write the next `<SYM>_*.csv`.
3. Print progress as plain stdout lines (the orchestrator wraps them as `LOG:`).
4. Register it in `orchestrator.py`'s `STEPS` list at the right position.
5. If it produces a new hand-off file, keep the `<SYMBOL>_` prefix convention.

## Add a research quant algorithm

Create `algorithms/<family>/<concept>/` with a `theory.tex` write-up beside a
`prototype.py`. For a fuller module, follow `volatility_forecasting`'s pattern: a
sub-package with `schemas.py` (typed results) and `run_*.py` entry scripts.

Feature selection across the ML pipeline reads
`factor_discovery/output/recommended_features.txt` — regenerate it via
`factor_discovery/factor_discovery.py` if you change the feature universe.

## After adding anything

- Add/adjust a mirror test in `tests/test_<area>.py` (pure functions, no network).
- Run the pipeline once for a ticker to confirm it flows: see the
  `retrain-pipeline` skill.
