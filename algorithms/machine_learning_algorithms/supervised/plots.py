"""
plots.py — All visualization functions for the supervised pipeline.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

from config import SYMBOL, OUT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


def _primary_metric(task, target):
    if task == "regression":
        return "ic"
    return "pr_auc" if target == "target_large_move" else "f1_w"


# ─────────────────────────────────────────────────────────────────────────────
# cross-validation plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_cv_primary(cv_df, target, task, version):
    """Primary metric across folds for all models (line chart)."""
    metric = _primary_metric(task, target)
    if metric not in cv_df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for model in cv_df["model"].unique():
        sub   = cv_df[cv_df["model"] == model]
        style = dict(linewidth=1.8, markersize=6)
        if model == "baseline":
            style.update(color="gray", linestyle="--", linewidth=1.2)
        ax.plot(sub["fold"], sub[metric], "o-", label=model, **style)
    ax.set_title(f"{SYMBOL} v{version} — {target} | {metric} (walk-forward CV)")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"v{version}_cv_{target}.png")


def plot_feat_freq(feat_counter, target, version, top_n=20):
    """Lasso feature selection frequency across folds (Version A only)."""
    if not feat_counter:
        return
    s = pd.Series(feat_counter).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, len(s) * 0.35)))
    s.plot(kind="barh", ax=ax, color="#1976D2")
    ax.set_xlabel("# folds selected by Lasso")
    ax.set_title(f"{SYMBOL} v{version} — {target} | Feature selection frequency")
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, f"v{version}_feat_freq_{target}.png")


# ─────────────────────────────────────────────────────────────────────────────
# holdout plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_holdout_bar(h_res, target, task, version):
    """Horizontal bar: primary metric for all models on the holdout set."""
    metric = _primary_metric(task, target)
    rows = [(m, res["metrics"].get(metric))
            for m, res in h_res.items()
            if "metrics" in res and res["metrics"].get(metric) is not None]
    if not rows:
        return
    models, vals = zip(*rows)
    colors = ["#9E9E9E" if m == "baseline" else "#1976D2" for m in models]
    fig, ax = plt.subplots(figsize=(7, max(3, len(rows) * 0.45)))
    ax.barh(models, vals, color=colors, height=0.6, edgecolor="white")
    bl = h_res.get("baseline", {}).get("metrics", {}).get(metric)
    if bl is not None:
        ax.axvline(bl, color="red", linestyle="--", linewidth=1.2,
                   label=f"baseline={bl:.3f}")
        ax.legend(fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title(f"{SYMBOL} v{version} — {target} | holdout {metric}")
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, f"v{version}_holdout_{target}.png")


def plot_pr_combined(h_res, target, version):
    """All models' PR curves on one chart (for the imbalanced binary target)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, res in h_res.items():
        if model == "baseline" or res.get("proba") is None:
            continue
        try:
            prec, rec, _ = precision_recall_curve(
                res["y_true"].astype(int), res["proba"][:, 1])
            ap = average_precision_score(
                res["y_true"].astype(int), res["proba"][:, 1])
            ax.plot(rec, prec, linewidth=1.5, label=f"{model} AP={ap:.3f}")
        except Exception:
            pass
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{SYMBOL} v{version} — {target} | PR curves (holdout)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"v{version}_pr_curves_{target}.png")


def plot_best_holdout(h_res, target, task, version):
    """Scatter (regression) or confusion matrix (classification) for best model."""
    metric = _primary_metric(task, target)
    best, best_val = None, -np.inf
    for model, res in h_res.items():
        if model == "baseline" or "metrics" not in res:
            continue
        v = res["metrics"].get(metric)
        if v is not None and v > best_val:
            best_val, best = v, model
    if best is None:
        return

    res      = h_res[best]
    y_t, y_p = res["y_true"], res["y_pred"]

    if task == "regression":
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_t, y_p, alpha=0.6, s=25, color="#E91E63", edgecolors="none")
        lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
        ax.plot(lims, lims, "k--", linewidth=0.8)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{SYMBOL} v{version} — {target} | best: {best} ({metric}={best_val:.3f})")
        plt.tight_layout()
        _save(fig, f"v{version}_best_{target}.png")
    else:
        cm = confusion_matrix(y_t.astype(int), y_p.astype(int))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{SYMBOL} v{version} — {target} | best: {best} ({metric}={best_val:.3f})")
        plt.tight_layout()
        _save(fig, f"v{version}_best_{target}.png")


def plot_feature_importance(h_res, target, task, version, top_n=20):
    """Native feature importances for the best holdout model (top-N bar chart)."""
    metric = _primary_metric(task, target)
    best, best_val = None, -np.inf
    for model, res in h_res.items():
        if model == "baseline" or "metrics" not in res:
            continue
        if res.get("importances") is None:
            continue
        v = res["metrics"].get(metric)
        if v is not None and v > best_val:
            best_val, best = v, model
    if best is None:
        return

    imp = h_res[best]["importances"]
    s = pd.Series(imp).sort_values(ascending=False).head(top_n)
    if s.empty:
        return

    fig, ax = plt.subplots(figsize=(9, max(4, len(s) * 0.42)))
    s.plot(kind="barh", ax=ax, color="#388E3C", edgecolor="white")
    ax.set_xlabel("Normalized importance")
    ax.set_title(
        f"{SYMBOL} v{version} — {target} | Feature importance  [{best}]  top-{top_n}"
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25, axis="x")
    plt.tight_layout()
    _save(fig, f"v{version}_feat_imp_{target}.png")


# ─────────────────────────────────────────────────────────────────────────────
# summary plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_version_comparison(all_holdout, targets, task):
    """Grouped bar: Version A vs B primary metric for each model and target."""
    key = "ic" if task == "regression" else "f1_w"
    rows = []
    for tgt in targets:
        for ver in ["A", "B"]:
            if (tgt, ver) not in all_holdout:
                continue
            for model, res in all_holdout[(tgt, ver)].items():
                if "error" in res or "metrics" not in res:
                    continue
                v = res["metrics"].get(key)
                if v is not None:
                    rows.append({"target": tgt, "version": ver, "model": model, key: v})
    if not rows:
        return

    df_p = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5), sharey=False)
    if len(targets) == 1:
        axes = [axes]
    for ax, tgt in zip(axes, targets):
        sub = df_p[df_p["target"] == tgt]
        if sub.empty:
            continue
        pivot = sub.pivot(index="model", columns="version", values=key)
        pivot.plot(kind="bar", ax=ax, color=["#1976D2", "#E91E63"], width=0.65)
        ax.set_title(tgt, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel(key)
        ax.legend(title="Version", fontsize=8)
        ax.tick_params(axis="x", labelrotation=40)
        ax.grid(True, alpha=0.2, axis="y")
    fig.suptitle(f"{SYMBOL} — Version A vs B  |  {key} (holdout)", fontsize=11)
    plt.tight_layout()
    _save(fig, f"version_AB_{task}.png")
