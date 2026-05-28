"""
factor_discovery.py — Feature selection and factor analysis.

Usage:
    python factor_discovery.py [SYMBOL]

Pipeline:
    1.  Load normalized features + targets
    2.  Correlation pruning   — drop features with pairwise corr > 0.95
    3.  Variance threshold    — drop near-zero variance features
    4.  Mutual information    — non-linear feature relevance vs each target
    5.  L1 (Lasso) selection  — sparse hard selection vs each regression target
    6.  L2 (Ridge) ranking    — soft importance ranking vs each regression target
    7.  L1 vs L2 comparison   — features robust across both regularizations
    8.  PCA                   — explained variance + component loadings
    9.  Summary               — final recommended feature set

Output (all saved to factor_discovery/output/):
    correlation_matrix.png
    pca_explained_variance.png
    pca_loadings.png
    mi_<target>.png            mutual information per target
    lasso_<target>.png         L1 selected features per target
    ridge_<target>.png         L2 top features per target
    l1_vs_l2_<target>.png      comparison per target
    factor_summary.csv         master table: feature × metric scores
    recommended_features.txt   final feature list
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")

# Config
SYMBOL   = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
ML_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR = ML_DIR / "data_pipelines"
OUT_DIR  = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

norm_csv    = DATA_DIR / f"{SYMBOL}_features_normalized.csv"
targets_csv = DATA_DIR / f"{SYMBOL}_targets.csv"

if not norm_csv.exists():
    print(f"ERROR: {norm_csv} not found — run normalize.py first.")
    sys.exit(1)

df_norm = pd.read_csv(norm_csv, index_col=0, parse_dates=True)
df_norm.index = pd.to_datetime(df_norm.index).tz_localize(None)

df_tgt = pd.read_csv(targets_csv, index_col=0, parse_dates=True)
df_tgt.index = pd.to_datetime(df_tgt.index).tz_localize(None)

TARGET_COLS = list(df_tgt.columns)
REG_TARGETS  = ["target_1d", "target_5d", "target_vol_5d"]
CLF_TARGETS  = ["target_dir_1d", "target_large_move", "target_regime"]

# Feature columns = everything in normalized CSV that isn't a target or meta
META_COLS = ["symbol", "fund_quarter_end", "opt_as_of"]
FEATURE_COLS = [
    c for c in df_norm.columns
    if c not in TARGET_COLS + META_COLS
    and pd.api.types.is_numeric_dtype(df_norm[c])
]

X_full = df_norm[FEATURE_COLS].copy()

print(f"=== factor_discovery: {SYMBOL} ===")
print(f"Features : {len(FEATURE_COLS)}")
print(f"Targets  : {TARGET_COLS}\n")
print(f"Saving plots to: {OUT_DIR}\n")

def save(fig, name):
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p.name}")

# Master score table
score_df = pd.DataFrame(index=FEATURE_COLS)

# 1. CORRELATION PRUNING
print("[1] Correlation pruning (threshold = 0.95)")

corr_matrix = X_full.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [c for c in upper.columns if any(upper[c] > 0.95)]

print(f"    Dropping {len(to_drop_corr)} highly correlated features:")
for c in to_drop_corr:
    partners = upper.index[upper[c] > 0.95].tolist()
    print(f"      ✗ {c:<30} corr>{0.95} with {partners}")

# Full correlation heatmap
fig, ax = plt.subplots(figsize=(20, 18))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, ax=ax, mask=mask, cmap="RdYlGn_r",
            vmin=0, vmax=1, linewidths=0.1,
            xticklabels=True, yticklabels=True)
ax.set_title(f"{SYMBOL} — Feature Correlation Matrix", fontweight="bold", fontsize=12)
ax.tick_params(labelsize=5)
fig.tight_layout()
save(fig, "correlation_matrix")

# Mark correlated features in score table
score_df["corr_dropped"] = score_df.index.isin(to_drop_corr).astype(int)

FEATURE_COLS_PRUNED = [c for c in FEATURE_COLS if c not in to_drop_corr]
X_pruned = X_full[FEATURE_COLS_PRUNED]
print(f"    Remaining features after pruning: {len(FEATURE_COLS_PRUNED)}")

# 2. VARIANCE THRESHOLD
print("\n[2] Variance threshold (< 0.01 → drop)")

variances = X_pruned.var()
low_var = variances[variances < 0.01].index.tolist()
if low_var:
    print(f"    Dropping {len(low_var)} near-zero variance features: {low_var}")
    X_pruned = X_pruned.drop(columns=low_var)
    score_df.loc[low_var, "corr_dropped"] = 1  # mark as removed
else:
    print("    None found.")

FEATURE_COLS_PRUNED = list(X_pruned.columns)
print(f"    Remaining features: {len(FEATURE_COLS_PRUNED)}")

# 3. MUTUAL INFORMATION
print("\n[3] Mutual information")

for target in REG_TARGETS + CLF_TARGETS:
    y = df_tgt[target].reindex(X_pruned.index).dropna()
    X_mi = X_pruned.reindex(y.index)

    is_clf = target in CLF_TARGETS
    mi_fn  = mutual_info_classif if is_clf else mutual_info_regression
    mi     = mi_fn(X_mi, y, random_state=42)
    mi_s   = pd.Series(mi, index=FEATURE_COLS_PRUNED).sort_values(ascending=False)

    score_df.loc[FEATURE_COLS_PRUNED, f"mi_{target}"] = mi_s

    # Plot top 25
    top25 = mi_s.head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top25.index[::-1], top25.values[::-1],
            color="#3498db", edgecolor="none")
    ax.set_xlabel("Mutual Information Score")
    ax.set_title(f"{SYMBOL} — Mutual Information vs {target}", fontweight="bold")
    fig.tight_layout()
    save(fig, f"mi_{target}")

    top5 = mi_s.head(5).index.tolist()
    print(f"    {target:<22}  top5: {top5}")

# 4. L1 — LASSO (regression targets only)
print("\n[4] L1 Lasso (hard selection)")

lasso_selected = {}

for target in REG_TARGETS:
    y = df_tgt[target].reindex(X_pruned.index).dropna()
    X_l = X_pruned.reindex(y.index)

    model = LassoCV(cv=5, max_iter=10000, random_state=42, n_alphas=50)
    model.fit(X_l, y)

    coefs = pd.Series(np.abs(model.coef_), index=FEATURE_COLS_PRUNED)
    nonzero = coefs[coefs > 0].sort_values(ascending=False)
    lasso_selected[target] = nonzero.index.tolist()

    score_df.loc[FEATURE_COLS_PRUNED, f"lasso_{target}"] = coefs

    print(f"    {target:<22}  alpha={model.alpha_:.5f}  "
          f"selected={len(nonzero)}/{len(FEATURE_COLS_PRUNED)} features")
    print(f"      top5: {nonzero.head(5).index.tolist()}")

    # Plot
    top_n = nonzero.head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in top_n.values]
    ax.barh(top_n.index[::-1], top_n.values[::-1], color="#e74c3c", edgecolor="none")
    ax.set_xlabel("|Lasso Coefficient|")
    ax.set_title(f"{SYMBOL} — L1 Lasso Selected Features: {target}", fontweight="bold")
    ax.axvline(0, color="white", linewidth=0.5)
    fig.tight_layout()
    save(fig, f"lasso_{target}")

# 5. L2 — RIDGE (regression targets only)
print("\n[5] L2 Ridge (soft ranking)")

ridge_top = {}
alphas = np.logspace(-3, 4, 50)

for target in REG_TARGETS:
    y = df_tgt[target].reindex(X_pruned.index).dropna()
    X_r = X_pruned.reindex(y.index)

    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_r, y)

    coefs = pd.Series(np.abs(model.coef_), index=FEATURE_COLS_PRUNED).sort_values(ascending=False)
    ridge_top[target] = coefs.head(25).index.tolist()

    score_df.loc[FEATURE_COLS_PRUNED, f"ridge_{target}"] = coefs

    print(f"    {target:<22}  alpha={model.alpha_:.4f}")
    print(f"      top5: {coefs.head(5).index.tolist()}")

    # Plot
    top25 = coefs.head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top25.index[::-1], top25.values[::-1], color="#2ecc71", edgecolor="none")
    ax.set_xlabel("|Ridge Coefficient|")
    ax.set_title(f"{SYMBOL} — L2 Ridge Feature Ranking: {target}", fontweight="bold")
    fig.tight_layout()
    save(fig, f"ridge_{target}")

# 6. L1 vs L2 COMPARISON
print("\n[6] L1 vs L2 comparison")

for target in REG_TARGETS:
    lasso_col = f"lasso_{target}"
    ridge_col = f"ridge_{target}"

    # Normalise both to [0,1] for fair comparison
    l1 = score_df[lasso_col].fillna(0)
    l2 = score_df[ridge_col].fillna(0)
    l1_norm = (l1 - l1.min()) / (l1.max() - l1.min() + 1e-12)
    l2_norm = (l2 - l2.min()) / (l2.max() - l2.min() + 1e-12)

    combined = ((l1_norm + l2_norm) / 2).sort_values(ascending=False)
    robust   = combined[combined > 0].head(20)

    score_df.loc[FEATURE_COLS_PRUNED, f"l1l2_combined_{target}"] = combined

    print(f"    {target}  — top features robust across L1+L2:")
    print(f"      {robust.head(10).index.tolist()}")

    # Scatter: L1 norm vs L2 norm
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(l1_norm, l2_norm, alpha=0.6, s=40)
    for feat in robust.head(12).index:
        ax.annotate(feat, (l1_norm[feat], l2_norm[feat]),
                    fontsize=6, alpha=0.85,
                    xytext=(4, 4), textcoords="offset points")
    ax.axline((0, 0), slope=1, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("L1 (Lasso) normalised |coef|")
    ax.set_ylabel("L2 (Ridge) normalised |coef|")
    ax.set_title(f"{SYMBOL} — L1 vs L2: {target}", fontweight="bold")
    fig.tight_layout()
    save(fig, f"l1_vs_l2_{target}")

# 7. PCA
print("\n[7] PCA")

X_pca = X_pruned.dropna()
pca   = PCA(n_components=min(len(FEATURE_COLS_PRUNED), len(X_pca)))
pca.fit(X_pca)

cum_var = np.cumsum(pca.explained_variance_ratio_)
n_90    = np.argmax(cum_var >= 0.90) + 1
n_95    = np.argmax(cum_var >= 0.95) + 1

print(f"    Components for 90% variance: {n_90}")
print(f"    Components for 95% variance: {n_95}")

# Explained variance plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 31), pca.explained_variance_ratio_[:30] * 100, color="#3498db")
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Explained Variance (%)")
axes[0].set_title("Scree Plot (top 30 components)")

axes[1].plot(range(1, len(cum_var) + 1), cum_var * 100, linewidth=1.5)
axes[1].axhline(90, color="orange", linestyle="--", linewidth=0.9, label="90%")
axes[1].axhline(95, color="red",    linestyle="--", linewidth=0.9, label="95%")
axes[1].axvline(n_90, color="orange", linestyle=":", linewidth=0.8)
axes[1].axvline(n_95, color="red",    linestyle=":", linewidth=0.8)
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance (%)")
axes[1].set_title("Cumulative Explained Variance")
axes[1].legend()
fig.suptitle(f"{SYMBOL} — PCA Explained Variance", fontweight="bold")
fig.tight_layout()
save(fig, "pca_explained_variance")

# Top feature loadings for first 5 components
loadings = pd.DataFrame(
    pca.components_[:5].T,
    index=FEATURE_COLS_PRUNED,
    columns=[f"PC{i+1}" for i in range(5)]
)
# Top 15 features by absolute loading magnitude across PC1-PC3
top_loading_feats = loadings[["PC1","PC2","PC3"]].abs().max(axis=1).nlargest(15).index

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for i, pc in enumerate(["PC1", "PC2", "PC3"]):
    data = loadings.loc[top_loading_feats, pc].sort_values()
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in data.values]
    axes[i].barh(data.index, data.values, color=colors, edgecolor="none")
    axes[i].axvline(0, color="white", linewidth=0.5)
    axes[i].set_title(f"{pc} ({pca.explained_variance_ratio_[i]*100:.1f}% var)")
    axes[i].tick_params(labelsize=8)
fig.suptitle(f"{SYMBOL} — PCA Component Loadings (top features)", fontweight="bold")
fig.tight_layout()
save(fig, "pca_loadings")

# Add PCA loading magnitudes to score table
for i, pc in enumerate(["PC1", "PC2", "PC3"]):
    score_df.loc[FEATURE_COLS_PRUNED, f"pca_{pc}_loading"] = loadings[pc].abs()

# 8. SUMMARY — recommended feature set
print("\n[8] Building recommended feature set")

# A feature is "recommended" if it clears at least 2 of these 3 bars:
#   - Appears in Lasso selection for at least 1 regression target
#   - Has MI score in top 30 for at least 1 target
#   - Has combined L1+L2 score > 0.2 for at least 1 target

lasso_any = pd.Series(False, index=FEATURE_COLS_PRUNED)
for target in REG_TARGETS:
    col = f"lasso_{target}"
    if col in score_df.columns:
        lasso_any = lasso_any | (score_df.loc[FEATURE_COLS_PRUNED, col].fillna(0) > 0)

mi_top30 = pd.Series(False, index=FEATURE_COLS_PRUNED)
for target in REG_TARGETS:
    col = f"mi_{target}"
    if col in score_df.columns:
        top30 = score_df.loc[FEATURE_COLS_PRUNED, col].nlargest(30).index
        mi_top30 = mi_top30 | pd.Series(FEATURE_COLS_PRUNED).isin(top30).values

l1l2_strong = pd.Series(False, index=FEATURE_COLS_PRUNED)
for target in REG_TARGETS:
    col = f"l1l2_combined_{target}"
    if col in score_df.columns:
        l1l2_strong = l1l2_strong | (score_df.loc[FEATURE_COLS_PRUNED, col].fillna(0) > 0.2)

votes = lasso_any.astype(int) + mi_top30.astype(int) + l1l2_strong.astype(int)
recommended = votes[votes >= 2].sort_values(ascending=False).index.tolist()

print(f"    Features clearing ≥2/3 criteria: {len(recommended)}")

# Save master score table
score_df.to_csv(OUT_DIR / "factor_summary.csv")
print(f"  ✓ factor_summary.csv")

# Save recommended feature list
rec_path = OUT_DIR / "recommended_features.txt"
rec_path.write_text("\n".join(recommended))
print(f"  ✓ recommended_features.txt  ({len(recommended)} features)")

print("\nRecommended features:")
for f in recommended:
    print(f"    {f}")

print(f"\n=== factor_discovery complete — outputs in {OUT_DIR} ===")
