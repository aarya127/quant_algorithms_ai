"""
metrics.py — Target-specific evaluation metrics (Phase 4).
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)


def reg_metrics(y_true, y_pred):
    """MAE, RMSE, R², directional accuracy, IC (Spearman rank correlation)."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[valid], y_pred[valid]
    if len(yt) < 2:
        return {}
    ic, _ = spearmanr(yt, yp)
    return {
        "mae":     float(mean_absolute_error(yt, yp)),
        "rmse":    float(np.sqrt(mean_squared_error(yt, yp))),
        "r2":      float(r2_score(yt, yp)),
        "dir_acc": float(np.mean(np.sign(yt) == np.sign(yp))),
        "ic":      float(ic) if not np.isnan(ic) else 0.0,
    }


def clf_metrics(y_true, y_pred, proba, target):
    """
    Accuracy, balanced accuracy, weighted/macro F1 for all classifiers.
    Extra metrics for target_large_move: precision, recall, PR-AUC, ROC-AUC.
    """
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    m = {
        "acc":      float(accuracy_score(yt, yp)),
        "bal_acc":  float(balanced_accuracy_score(yt, yp)),
        "f1_w":     float(f1_score(yt, yp, average="weighted",  zero_division=0)),
        "f1_macro": float(f1_score(yt, yp, average="macro",     zero_division=0)),
    }
    if target == "target_large_move":
        m["precision"] = float(precision_score(yt, yp, zero_division=0))
        m["recall"]    = float(recall_score(yt, yp,    zero_division=0))
        if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
            try:
                m["pr_auc"]  = float(average_precision_score(yt, proba[:, 1]))
                m["roc_auc"] = float(roc_auc_score(yt, proba[:, 1]))
            except Exception:
                pass
    elif proba is not None:
        n_cls = len(np.unique(yt))
        try:
            if n_cls == 2:
                m["roc_auc"] = float(roc_auc_score(yt, proba[:, 1]))
            else:
                m["roc_auc"] = float(roc_auc_score(
                    yt, proba, multi_class="ovr", average="weighted"))
        except Exception:
            pass
    return m
