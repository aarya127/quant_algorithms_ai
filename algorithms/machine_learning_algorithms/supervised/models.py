"""
models.py — Model factories and feature importance extraction (Phase 3).
"""
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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
