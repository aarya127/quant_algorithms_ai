"""
models.py — Model factories, feature importance, and hyperparameter tuning (Stage 7C).
"""
import numpy as np
from scipy.stats import loguniform, uniform
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

from config import BINARY_CLF


def reg_model_set():
    """Regression model ladder: Ridge, ElasticNet, Huber, RF, XGB, LGB."""
    return {
        "ridge":      Ridge(alpha=10.0),
        "elasticnet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "huber":      HuberRegressor(epsilon=1.5, max_iter=500),
        "rf":         RandomForestRegressor(n_estimators=200, max_depth=6,
                                            random_state=42, n_jobs=1),
        "xgb":        xgb.XGBRegressor(n_estimators=300, max_depth=4,
                                        learning_rate=0.05, subsample=0.8,
                                        colsample_bytree=0.8, reg_alpha=0.1,
                                        reg_lambda=1.0, random_state=42,
                                        verbosity=0, n_jobs=1),
        "lgb":        lgb.LGBMRegressor(n_estimators=300, max_depth=4,
                                         learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, reg_alpha=0.1,
                                         reg_lambda=1.0, random_state=42,
                                         verbosity=-1, n_jobs=1),
    }


def clf_model_set(n_cls, is_binary_imb, spw, is_multi):
    """Classification model ladder: Logistic, CalibratedSVC, RF, XGB, LGB."""
    return {
        "logistic": LogisticRegression(C=1.0, class_weight="balanced",
                                       max_iter=2000, solver="lbfgs",
                                       random_state=42),
        "svc_cal":  CalibratedClassifierCV(
                        LinearSVC(class_weight="balanced", max_iter=5000,
                                  random_state=42), cv=3),
        "rf":       RandomForestClassifier(n_estimators=200, max_depth=6,
                                           class_weight="balanced",
                                           random_state=42, n_jobs=1),
        "xgb":      xgb.XGBClassifier(
                        n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0,
                        objective="multi:softprob" if n_cls > 2 else "binary:logistic",
                        num_class=n_cls if n_cls > 2 else None,
                        scale_pos_weight=spw if (is_binary_imb and n_cls == 2) else 1.0,
                        random_state=42, verbosity=0, n_jobs=1,
                        eval_metric="mlogloss" if n_cls > 2 else "logloss"),
        "lgb":      lgb.LGBMClassifier(
                        n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0,
                        objective="multiclass" if n_cls > 2 else "binary",
                        num_class=n_cls if n_cls > 2 else None,
                        class_weight="balanced",
                        random_state=42, verbosity=-1, n_jobs=1),
    }


def _get_feature_importances(model, name, sel_features):
    """
    Return {feature: normalized_importance} for supported model types.

    - Tree models (rf, xgb, lgb): model.feature_importances_
    - Linear models (ridge, elasticnet, huber): |coef_|
    - Logistic: mean |coef_| across classes
    - svc_cal: skipped (calibrated SVC internals are complex)
    """
    try:
        if name in ("rf", "xgb", "lgb"):
            fi = np.array(model.feature_importances_, dtype=float)
        elif name in ("ridge", "elasticnet", "huber"):
            fi = np.abs(np.array(model.coef_, dtype=float))
        elif name == "logistic":
            coef = np.array(model.coef_, dtype=float)
            fi = np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(coef[0])
        else:
            return None
        if len(fi) != len(sel_features):
            return None
        total = fi.sum()
        if total > 0:
            fi = fi / total
        return dict(zip(sel_features, fi.tolist()))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7C — Hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────────────
def reg_param_dists():
    """RandomizedSearch parameter distributions for regression models."""
    return {
        "ridge":      {"alpha": loguniform(0.1, 100)},
        "elasticnet": {"alpha": loguniform(1e-3, 1), "l1_ratio": uniform(0.1, 0.85)},
        "huber":      {"epsilon": uniform(1.05, 1.45), "alpha": loguniform(1e-5, 0.1)},
        "rf":         {"n_estimators": [100, 200, 300],
                       "max_depth": [4, 6, 8, None],
                       "min_samples_leaf": [1, 2, 4]},
        "xgb":        {"n_estimators": [100, 200, 300], "max_depth": [3, 4, 6],
                       "learning_rate": loguniform(0.01, 0.3),
                       "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
                       "reg_alpha": loguniform(0.01, 1), "reg_lambda": loguniform(0.1, 10)},
        "lgb":        {"n_estimators": [100, 200, 300], "max_depth": [3, 4, 6],
                       "learning_rate": loguniform(0.01, 0.3),
                       "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
                       "reg_alpha": loguniform(0.01, 1), "reg_lambda": loguniform(0.1, 10)},
    }


def clf_param_dists():
    """RandomizedSearch parameter distributions for classification models."""
    return {
        "logistic": {"C": loguniform(0.01, 100)},
        "svc_cal":  {"estimator__C": loguniform(0.01, 100)},
        "rf":       {"n_estimators": [100, 200, 300],
                     "max_depth": [4, 6, 8, None],
                     "min_samples_leaf": [1, 2, 4]},
        "xgb":      {"n_estimators": [100, 200, 300], "max_depth": [3, 4, 6],
                     "learning_rate": loguniform(0.01, 0.3),
                     "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
                     "reg_alpha": loguniform(0.01, 1), "reg_lambda": loguniform(0.1, 10)},
        "lgb":      {"n_estimators": [100, 200, 300], "max_depth": [3, 4, 6],
                     "learning_rate": loguniform(0.01, 0.3),
                     "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
                     "reg_alpha": loguniform(0.01, 1), "reg_lambda": loguniform(0.1, 10)},
    }


def _tune_scoring(task, target):
    """Return the sklearn scoring string for inner CV during tuning."""
    if task == "regression":
        return "r2"
    return "average_precision" if target in BINARY_CLF else "f1_weighted"


def tune_model(name, model, param_dist, X, y, scoring, n_iter, n_splits=3):
    """
    RandomizedSearchCV with TimeSeriesSplit inner folds (no data leakage).

    Returns (best_estimator, best_params) on success.
    On failure prints a warning, falls back to fitting with original params,
    and returns (model, {}).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    try:
        rs = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=tscv,
            refit=True,
            random_state=42,
            n_jobs=1,
            error_score="raise",
        )
        rs.fit(X, y)
        return rs.best_estimator_, rs.best_params_
    except Exception as e:
        print(f"      [tune {name} failed: {e}]")
        try:
            model.fit(X, y)
        except Exception:
            pass
        return model, {}
