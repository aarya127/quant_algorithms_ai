"""
unsupervised.py  —  Regime clustering + anomaly detection
=========================================================
Usage:
    python algorithms/machine_learning_algorithms/unsupervised/unsupervised.py NVDA

Pipeline
--------
1. Temporal train/test split  (80 / 20 by date)
2. PCA on train → transform all rows  (16 components = 90 % variance)
3. Elbow + silhouette  (k = 2 … 6) → pick best k
4. K-Means fit on train-PCA → label all rows
5. GMM fit on train-PCA → BIC curve, confirm k
6. Isolation Forest fit on train-raw-34 → anomaly labels all rows
7. Save  NVDA_features_with_regimes.csv
8. Plots  (all → unsupervised/output/)
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
ROOT       = HERE.parent                                          # machine_learning_algorithms/
PIPELINES  = ROOT / "data_pipelines"
FD_OUTPUT  = ROOT / "factor_discovery" / "output"
OUT_DIR    = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "NVDA"

NORM_CSV  = PIPELINES / f"{SYMBOL}_features_normalized.csv"
FEAT_FILE = FD_OUTPUT  / "recommended_features.txt"

# ── config ────────────────────────────────────────────────────────────────────
TRAIN_FRAC      = 0.80
PCA_COMPONENTS  = 16          # 90 % variance from factor_discovery
K_RANGE         = range(2, 7)
IF_CONTAMINATION = 0.05       # expected fraction of anomalies


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 0. load
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== unsupervised: {SYMBOL} ===")

df = pd.read_csv(NORM_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
features = [ln.strip() for ln in FEAT_FILE.read_text().splitlines() if ln.strip()]

# keep only features that are actually in the CSV
features = [f for f in features if f in df.columns]
print(f"Rows      : {len(df)}")
print(f"Features  : {len(features)}")
print(f"Saving to : {OUT_DIR}\n")

X_all  = df[features].values.astype(float)
dates  = df["Date"].values

# ─────────────────────────────────────────────────────────────────────────────
# 1. temporal train / test split
# ─────────────────────────────────────────────────────────────────────────────
n_train = int(len(df) * TRAIN_FRAC)
n_test  = len(df) - n_train

X_train = X_all[:n_train]
X_test  = X_all[n_train:]
dates_train = dates[:n_train]
dates_test  = dates[n_train:]

print(f"[1] Temporal split")
print(f"    Train : {n_train} rows  ({pd.Timestamp(dates_train[0]).date()} → {pd.Timestamp(dates_train[-1]).date()})")
print(f"    Test  : {n_test}  rows  ({pd.Timestamp(dates_test[0]).date()} → {pd.Timestamp(dates_test[-1]).date()})")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PCA — fit on train, transform all
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2] PCA ({PCA_COMPONENTS} components → ~90 % variance, fit on train)")

pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
pca.fit(X_train)

Z_all   = pca.transform(X_all)
Z_train = Z_all[:n_train]
Z_test  = Z_all[n_train:]

cum_var = np.cumsum(pca.explained_variance_ratio_)
print(f"    Variance captured: {cum_var[-1]*100:.1f}%")

# plot: cumulative explained variance
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, PCA_COMPONENTS + 1), cum_var * 100, "o-", color="#2196F3")
ax.axhline(90, color="red", linestyle="--", linewidth=0.8, label="90 %")
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative explained variance (%)")
ax.set_title(f"{SYMBOL} — PCA cumulative variance (train set)")
ax.legend()
ax.grid(True, alpha=0.3)
_save(fig, "pca_variance_train.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. elbow + silhouette → choose k
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] K-Means elbow + silhouette (fit on train PCA)")

inertias, silhouettes = [], []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Z_train)
    inertias.append(km.inertia_)
    sil = silhouette_score(Z_train, labels) if k > 1 else float("nan")
    silhouettes.append(sil)
    print(f"    k={k}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

best_k = list(K_RANGE)[int(np.argmax(silhouettes))]
print(f"    → Best k by silhouette: {best_k}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ks = list(K_RANGE)
axes[0].plot(ks, inertias, "o-", color="#E91E63")
axes[0].set_title("Elbow — inertia")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].grid(True, alpha=0.3)

axes[1].plot(ks, silhouettes, "o-", color="#4CAF50")
axes[1].axvline(best_k, color="red", linestyle="--", linewidth=0.8, label=f"best k={best_k}")
axes[1].set_title("Silhouette score")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle(f"{SYMBOL} — K-Means selection", fontsize=12)
plt.tight_layout()
_save(fig, "kmeans_selection.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. K-Means final fit (best_k) on train → label all
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4] K-Means (k={best_k}) — fit on train, label all rows")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
km_final.fit(Z_train)

km_labels_all = km_final.predict(Z_all)
df["cluster_kmeans"] = km_labels_all

# relabel clusters by mean realized_vol so 0=low-vol, N-1=high-vol (interpretable)
if "realized_vol_20d" in df.columns:
    cluster_vol = df.groupby("cluster_kmeans")["realized_vol_20d"].mean().sort_values()
    remap = {old: new for new, old in enumerate(cluster_vol.index)}
    df["cluster_kmeans"] = df["cluster_kmeans"].map(remap)
    km_labels_all = df["cluster_kmeans"].values

train_dist = pd.Series(km_labels_all[:n_train]).value_counts().sort_index()
print(f"    Train cluster distribution: {train_dist.to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. GMM — BIC over k, final model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] GMM BIC curve (fit on train PCA)")

bics = []
for k in K_RANGE:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=5)
    gmm.fit(Z_train)
    bics.append(gmm.bic(Z_train))
    print(f"    k={k}  BIC={gmm.bic(Z_train):,.1f}")

best_k_gmm = list(K_RANGE)[int(np.argmin(bics))]
print(f"    → Best k by BIC: {best_k_gmm}")

# use the same k as K-Means for consistency (take the lower BIC-preferred k if they differ)
k_final = min(best_k, best_k_gmm)
print(f"    → Using k={k_final} for GMM (min of silhouette-best and BIC-best)")

gmm_final = GaussianMixture(n_components=k_final, covariance_type="full", random_state=42, n_init=20)
gmm_final.fit(Z_train)
gmm_labels_all = gmm_final.predict(Z_all)
df["cluster_gmm"] = gmm_labels_all

# same vol-based relabeling for GMM
if "realized_vol_20d" in df.columns:
    cluster_vol_gmm = df.groupby("cluster_gmm")["realized_vol_20d"].mean().sort_values()
    remap_gmm = {old: new for new, old in enumerate(cluster_vol_gmm.index)}
    df["cluster_gmm"] = df["cluster_gmm"].map(remap_gmm)

# GMM posterior probabilities (uncertainty measure)
proba = gmm_final.predict_proba(Z_all)
for i in range(k_final):
    df[f"gmm_prob_regime_{i}"] = proba[:, i]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(K_RANGE), bics, "o-", color="#FF9800")
ax.axvline(best_k_gmm, color="red", linestyle="--", linewidth=0.8, label=f"best k={best_k_gmm}")
ax.set_title(f"{SYMBOL} — GMM BIC")
ax.set_xlabel("k")
ax.set_ylabel("BIC")
ax.legend()
ax.grid(True, alpha=0.3)
_save(fig, "gmm_bic.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Isolation Forest — fit on train raw features → label all
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[6] Isolation Forest (contamination={IF_CONTAMINATION}) — fit on train raw features")

iso = IsolationForest(contamination=IF_CONTAMINATION, random_state=42, n_estimators=200)
iso.fit(X_train)

iso_labels_all  = iso.predict(X_all)     # +1 normal, -1 anomaly
iso_scores_all  = iso.score_samples(X_all)  # lower = more anomalous

df["anomaly_iso"] = (iso_labels_all == -1).astype(int)   # 1 = anomaly
df["anomaly_score"] = iso_scores_all

n_anomaly_train = (iso_labels_all[:n_train] == -1).sum()
n_anomaly_test  = (iso_labels_all[n_train:] == -1).sum()
print(f"    Anomalies in train: {n_anomaly_train} / {n_train}  ({n_anomaly_train/n_train*100:.1f}%)")
print(f"    Anomalies in test : {n_anomaly_test} / {n_test}  ({n_anomaly_test/n_test*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. save output CSV
# ─────────────────────────────────────────────────────────────────────────────
out_csv = PIPELINES / f"{SYMBOL}_features_with_regimes.csv"
df.to_csv(out_csv, index=False)
print(f"\n[7] Saved → {out_csv.name}  ({len(df)} rows × {len(df.columns)} cols)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. plots
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Generating plots")

dates_dt = pd.to_datetime(dates)

# ── 8a. PCA 2D scatter — K-Means clusters ────────────────────────────────────
pca2d = PCA(n_components=2, random_state=42)
pca2d.fit(X_train)
Z2d_all = pca2d.transform(X_all)

palette = sns.color_palette("tab10", best_k)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for k, ax in zip(["cluster_kmeans", "cluster_gmm"], axes):
    labels_plot = df[k].values
    n_k = df[k].nunique()
    pal = sns.color_palette("tab10", n_k)
    for c in range(n_k):
        mask = labels_plot == c
        ax.scatter(Z2d_all[mask, 0], Z2d_all[mask, 1],
                   c=[pal[c]], label=f"Regime {c}", alpha=0.7, s=30, edgecolors="none")
    # mark anomalies
    anom_mask = df["anomaly_iso"].values == 1
    ax.scatter(Z2d_all[anom_mask, 0], Z2d_all[anom_mask, 1],
               marker="x", color="red", s=60, linewidths=1.2, label="Anomaly", zorder=5)
    # train / test boundary
    ax.axvline(Z2d_all[n_train - 1, 0], color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{SYMBOL} — {k.replace('cluster_', '').upper()} clusters (PCA 2D)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
_save(fig, "pca2d_clusters.png")

# ── 8b. regime timeline — K-Means ────────────────────────────────────────────
km_colors = sns.color_palette("tab10", best_k)
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

# top: price (normalized Open as proxy)
ax0 = axes[0]
ax0.plot(dates_dt, df["Open"].values, color="#1565C0", linewidth=1)
ax0.set_ylabel("Normalized price (Open)")
ax0.set_title(f"{SYMBOL} — Regime timeline (K-Means k={best_k})")
ax0.grid(True, alpha=0.2)

# shade by regime
km_lab = df["cluster_kmeans"].values
for i in range(len(km_lab) - 1):
    ax0.axvspan(dates_dt[i], dates_dt[i + 1], alpha=0.15,
                color=km_colors[km_lab[i]], linewidth=0)
# anomaly ticks
for i, (is_anom, d) in enumerate(zip(df["anomaly_iso"].values, dates_dt)):
    if is_anom:
        ax0.axvline(d, color="red", linewidth=0.8, alpha=0.6)

# bottom: cluster label as step plot
ax1 = axes[1]
for c in range(best_k):
    mask = km_lab == c
    ax1.scatter(dates_dt[mask], np.zeros(mask.sum()) + c,
                color=km_colors[c], s=25, label=f"Regime {c}", alpha=0.9)
ax1.set_yticks(range(best_k))
ax1.set_yticklabels([f"Regime {c}" for c in range(best_k)])
ax1.set_ylabel("Cluster")
ax1.set_xlabel("Date")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax1.grid(True, alpha=0.2)

# train/test split line
split_date = dates_dt[n_train]
for ax in axes:
    ax.axvline(split_date, color="black", linestyle="--", linewidth=1,
               label="train|test split" if ax is axes[0] else "")
if best_k <= 4:
    axes[1].legend(fontsize=8, loc="upper left")

plt.tight_layout()
_save(fig, "regime_timeline_kmeans.png")

# ── 8c. GMM probability bands ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
bottom = np.zeros(len(df))
gmm_pal = sns.color_palette("tab10", k_final)
for c in range(k_final):
    col = f"gmm_prob_regime_{c}"
    vals = df[col].values
    ax.fill_between(dates_dt, bottom, bottom + vals,
                    alpha=0.75, color=gmm_pal[c], label=f"Regime {c} prob.")
    bottom += vals

ax.axvline(split_date, color="black", linestyle="--", linewidth=1, label="train|test split")
ax.set_ylim(0, 1)
ax.set_ylabel("GMM posterior probability")
ax.set_title(f"{SYMBOL} — GMM regime posterior probabilities")
ax.legend(fontsize=8, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.grid(True, alpha=0.2)
_save(fig, "gmm_probabilities.png")

# ── 8d. anomaly score over time ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
scores = df["anomaly_score"].values
ax.plot(dates_dt, scores, color="#607D8B", linewidth=0.8, alpha=0.8)

# highlight anomaly days
anom_dates  = dates_dt[df["anomaly_iso"].values == 1]
anom_scores = scores[df["anomaly_iso"].values == 1]
ax.scatter(anom_dates, anom_scores, color="red", s=40, zorder=5, label="Anomaly")

ax.axvline(split_date, color="black", linestyle="--", linewidth=1, label="train|test split")
ax.set_ylabel("Isolation Forest score (lower = more anomalous)")
ax.set_title(f"{SYMBOL} — Anomaly detection (Isolation Forest)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.grid(True, alpha=0.2)
_save(fig, "anomaly_score.png")

# ── 8e. regime stats summary ──────────────────────────────────────────────────
print("\n[9] Regime statistics (K-Means, full dataset)")
stat_cols = ["realized_vol_20d", "daily_return", "vix_level", "volume_zscore"]
stat_cols = [c for c in stat_cols if c in df.columns]

regime_stats = df.groupby("cluster_kmeans")[stat_cols].mean()
print(regime_stats.to_string())

# heatmap of regime means
fig, ax = plt.subplots(figsize=(8, max(3, best_k * 1.2)))
sns.heatmap(regime_stats, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
            linewidths=0.5, ax=ax)
ax.set_title(f"{SYMBOL} — Regime mean feature values (K-Means k={best_k})")
ax.set_ylabel("Regime (cluster)")
plt.tight_layout()
_save(fig, "regime_stats_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# done
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== unsupervised complete — outputs in {OUT_DIR} ===")
print(f"    {SYMBOL}_features_with_regimes.csv ready for supervised modeling")
