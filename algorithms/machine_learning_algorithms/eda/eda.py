"""
eda.py — Exploratory Data Analysis for the NVDA feature matrix.

Usage:
    python eda.py [SYMBOL]

Sections:
    1.  Overview          — shape, dtypes, basic stats
    2.  Target variable   — next-day log return distribution & stationarity
    3.  Price & volume    — close price + volume over time
    4.  Return analysis   — distribution, QQ-plot, autocorrelation
    5.  Volatility        — realized vol regimes over time
    6.  Technical signals — RSI, MACD, BB position over time
    7.  Momentum          — 1m / 3m / 6m / 12m momentum heatmap
    8.  Sentiment         — AV + Marketaux daily sentiment vs. price
    9.  Macro / cross-asset — VIX, yield curve, SPY beta
    10. Fundamentals      — quarterly fundamental ratios over time
    11. Correlation matrix — top correlated features with next-day return
    12. Feature distributions — histograms for every numeric feature

Output:
    algorithms/machine_learning_algorithms/eda_output/   (PNG files)
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — saves to files instead of opening windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Config
SYMBOL  = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
HERE     = Path(__file__).parent
DATA_DIR = HERE.parent / "data_pipelines"
OUT_DIR  = HERE

CLEAN_CSV = DATA_DIR / f"{SYMBOL}_features_clean.csv"
if not CLEAN_CSV.exists():
    print(f"ERROR: {CLEAN_CSV} not found. Run run_pipeline.py then clean.py first.")
    sys.exit(1)

sns.set_theme(style="darkgrid", palette="muted")
FIGSIZE = (14, 5)

df = pd.read_csv(CLEAN_CSV, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index).tz_localize(None)
df.index.name = "Date"

# Define the target: next-day log return
df["target_1d"] = df["log_return"].shift(-1)

print(f"Loaded {SYMBOL}: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
print(f"Saving plots to: {OUT_DIR}\n")

# Helper
def save(fig, name: str):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# 1. OVERVIEW
print("\n[1] Overview")

numeric = df.select_dtypes(include="number")
desc = numeric.describe().T
desc["skew"]     = numeric.skew()
desc["kurtosis"] = numeric.kurtosis()
desc["null_pct"] = (df.isnull().mean() * 100).round(2)

print(desc[["count","mean","std","min","max","skew","kurtosis","null_pct"]].to_string())


# 2. TARGET VARIABLE — next-day return
print("\n[2] Target variable")

tgt = df["target_1d"].dropna()
print(f"  mean={tgt.mean():.5f}  std={tgt.std():.5f}  skew={tgt.skew():.3f}  kurt={tgt.kurtosis():.3f}")

stat, p = stats.shapiro(tgt.sample(min(len(tgt), 5000), random_state=42))
print(f"  Shapiro-Wilk: W={stat:.4f}  p={p:.4e}  {'→ NOT normal' if p < 0.05 else '→ normal'}")

adf_result = None
try:
    from statsmodels.tsa.stattools import adfuller
    adf = adfuller(tgt, autolag="AIC")
    print(f"  ADF test:  stat={adf[0]:.4f}  p={adf[1]:.4e}  {'→ stationary ✓' if adf[1] < 0.05 else '→ non-stationary!'}")
    adf_result = adf
except ImportError:
    print("  (statsmodels not installed — skipping ADF test)")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(f"{SYMBOL} — Target: Next-Day Log Return", fontweight="bold")

axes[0].hist(tgt, bins=50, edgecolor="white", linewidth=0.3)
axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
axes[0].set_title("Distribution")
axes[0].set_xlabel("Log Return")

stats.probplot(tgt, dist="norm", plot=axes[1])
axes[1].set_title("QQ Plot vs Normal")

axes[2].plot(tgt.index, tgt.values, linewidth=0.7, alpha=0.8)
axes[2].axhline(0, color="red", linestyle="--", linewidth=0.8)
axes[2].set_title("Returns Over Time")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
fig.autofmt_xdate()

save(fig, "02_target_return")


# 3. PRICE & VOLUME
print("\n[3] Price & volume")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle(f"{SYMBOL} — Price & Volume", fontweight="bold")

ax1.plot(df.index, df["Close"], linewidth=1.2, label="Close")
ax1.plot(df.index, df["SMA_20"],  linewidth=0.8, linestyle="--", label="SMA 20",  alpha=0.7)
ax1.plot(df.index, df["SMA_50"],  linewidth=0.8, linestyle="--", label="SMA 50",  alpha=0.7)
ax1.plot(df.index, df["SMA_200"], linewidth=0.8, linestyle=":",  label="SMA 200", alpha=0.7)
ax1.fill_between(df.index, df["BB_lower"], df["BB_upper"], alpha=0.1, label="BB bands")
ax1.set_ylabel("Price (USD)")
ax1.legend(fontsize=8)

ax2.bar(df.index, df["Volume"], width=1, alpha=0.6)
ax2.set_ylabel("Volume")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
fig.autofmt_xdate()

save(fig, "03_price_volume")


# 4. RETURN ANALYSIS — ACF/PACF + rolling vol
print("\n[4] Return analysis")

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.suptitle(f"{SYMBOL} — Autocorrelation of Log Returns", fontweight="bold")
    plot_acf(df["log_return"].dropna(),  ax=axes[0], lags=40, title="ACF")
    plot_pacf(df["log_return"].dropna(), ax=axes[1], lags=40, title="PACF", method="ywm")
    save(fig, "04a_acf_pacf")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.suptitle(f"{SYMBOL} — Autocorrelation of |Returns| (Volatility Clustering)", fontweight="bold")
    plot_acf(df["log_return"].abs().dropna(),  ax=axes[0], lags=40, title="ACF |returns|")
    plot_pacf(df["log_return"].abs().dropna(), ax=axes[1], lags=40, title="PACF |returns|", method="ywm")
    save(fig, "04b_acf_abs_returns")

except ImportError:
    print("  (statsmodels not installed — skipping ACF/PACF)")


# 5. VOLATILITY REGIMES
print("\n[5] Volatility")

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
fig.suptitle(f"{SYMBOL} — Volatility Features", fontweight="bold")

axes[0].plot(df.index, df["realized_vol_20d"] * 100, label="Realized Vol 20d", linewidth=1)
axes[0].plot(df.index, df["realized_vol_60d"] * 100, label="Realized Vol 60d", linewidth=1, linestyle="--")
axes[0].set_ylabel("Ann. Vol (%)")
axes[0].legend(fontsize=8)

axes[1].plot(df.index, df["vol_of_vol"] * 100, color="purple", linewidth=0.9)
axes[1].set_ylabel("Vol-of-Vol (%)")

axes[2].plot(df.index, df["max_drawdown_20d"] * 100, color="crimson", linewidth=0.9)
axes[2].fill_between(df.index, df["max_drawdown_20d"] * 100, 0, alpha=0.2, color="crimson")
axes[2].set_ylabel("Max Drawdown 20d (%)")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
fig.autofmt_xdate()

save(fig, "05_volatility")


# 6. TECHNICAL SIGNALS
print("\n[6] Technical signals")

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle(f"{SYMBOL} — Technical Indicators", fontweight="bold")

axes[0].plot(df.index, df["RSI_14"], linewidth=0.9)
axes[0].axhline(70, color="red",   linestyle="--", linewidth=0.8, alpha=0.7, label="Overbought 70")
axes[0].axhline(30, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label="Oversold 30")
axes[0].set_ylabel("RSI 14")
axes[0].set_ylim(0, 100)
axes[0].legend(fontsize=8)

axes[1].plot(df.index, df["MACD"],        linewidth=0.9, label="MACD")
axes[1].plot(df.index, df["MACD_signal"], linewidth=0.9, linestyle="--", label="Signal")
axes[1].bar(df.index, df["MACD_hist"], width=1, alpha=0.4, label="Histogram")
axes[1].axhline(0, color="white", linewidth=0.5)
axes[1].set_ylabel("MACD")
axes[1].legend(fontsize=8)

axes[2].plot(df.index, df["bb_pct"], linewidth=0.9)
axes[2].axhline(1.0, color="red",   linestyle="--", linewidth=0.8, alpha=0.7)
axes[2].axhline(0.0, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
axes[2].set_ylabel("BB %B")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
fig.autofmt_xdate()

save(fig, "06_technical_signals")


# 7. MOMENTUM
print("\n[7] Momentum")

mom_cols = ["reversal_1w", "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m"]
mom_df   = df[mom_cols] * 100   # to percent

fig, axes = plt.subplots(len(mom_cols), 1, figsize=(14, 10), sharex=True)
fig.suptitle(f"{SYMBOL} — Momentum Signals", fontweight="bold")

for ax, col in zip(axes, mom_cols):
    vals = mom_df[col]
    ax.bar(df.index, vals, width=1,
           color=np.where(vals >= 0, "#2ecc71", "#e74c3c"), alpha=0.7)
    ax.axhline(0, color="white", linewidth=0.5)
    ax.set_ylabel(col.replace("_", " "), fontsize=8)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
fig.autofmt_xdate()
save(fig, "07_momentum")


# 8. SENTIMENT vs. PRICE
print("\n[8] Sentiment")

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle(f"{SYMBOL} — News Sentiment vs. Price", fontweight="bold")

ax_price = axes[0]
ax_price.plot(df.index, df["Close"], linewidth=1, color="#3498db")
ax_price.set_ylabel("Close (USD)")

ax_av = axes[1]
ax_av.plot(df.index, df["news_sent_av_7d"],        linewidth=1,   label="AV 7d",         color="steelblue")
ax_av.plot(df.index, df["news_sent_marketaux_7d"], linewidth=1,   label="Marketaux 7d",  color="darkorange")
ax_av.plot(df.index, df["news_sent_score"],        linewidth=0.6, label="Blended score", color="gray", alpha=0.6)
ax_av.axhline(0, color="white", linewidth=0.5)
ax_av.set_ylabel("Sentiment Score")
ax_av.legend(fontsize=8)

ax_cnt = axes[2]
ax_cnt.bar(df.index, df["news_count"], width=1, alpha=0.7, color="slateblue")
ax_cnt.set_ylabel("Article Count")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
fig.autofmt_xdate()

save(fig, "08_sentiment")


# 9. MACRO / CROSS-ASSET
print("\n[9] Macro / cross-asset")

fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
fig.suptitle(f"{SYMBOL} — Macro & Cross-Asset Features", fontweight="bold")

axes[0].plot(df.index, df["vix_level"], linewidth=0.9, color="crimson")
axes[0].axhline(20, color="orange", linestyle="--", linewidth=0.8, alpha=0.7, label="VIX 20")
axes[0].set_ylabel("VIX Level")
axes[0].legend(fontsize=8)

axes[1].plot(df.index, df["yield_10y"] * 100,         linewidth=0.9, label="10Y Yield")
axes[1].plot(df.index, df["yield_curve_slope"] * 100, linewidth=0.9, linestyle="--", label="Curve Slope (10Y-2Y)")
axes[1].axhline(0, color="white", linewidth=0.5)
axes[1].set_ylabel("Yield (%)")
axes[1].legend(fontsize=8)

axes[2].plot(df.index, df["spy_beta_60d"], linewidth=0.9, color="teal")
axes[2].axhline(1, color="white", linestyle="--", linewidth=0.5)
axes[2].set_ylabel("SPY Beta 60d")

axes[3].plot(df.index, df["qqq_corr_20d"],   linewidth=0.9, label="QQQ Corr 20d")
axes[3].plot(df.index, df["sox_rel_strength"], linewidth=0.9, linestyle="--", label="SOX Rel Strength")
axes[3].axhline(0, color="white", linewidth=0.5)
axes[3].set_ylabel("Correlation / Rel Strength")
axes[3].legend(fontsize=8)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
fig.autofmt_xdate()

save(fig, "09_macro")


# 10. FUNDAMENTALS
print("\n[10] Fundamentals")

fund_pairs = [
    ("fund_gross_margin",    "fund_net_margin",    "Margins"),
    ("fund_roa",             "fund_roe",           "ROA vs ROE"),
    ("fund_debt_to_equity",  "fund_current_ratio", "Leverage & Liquidity"),
    ("fund_eps_ttm",         "fund_fcf_ttm",       "EPS (TTM) & FCF (TTM)"),
]

fig, axes = plt.subplots(len(fund_pairs), 1, figsize=(14, 11), sharex=True)
fig.suptitle(f"{SYMBOL} — Fundamentals Over Time", fontweight="bold")

for ax, (col1, col2, title) in zip(axes, fund_pairs):
    ax2 = ax.twinx()
    ax.plot(df.index,  df[col1], linewidth=1,   label=col1.replace("fund_",""), color="#3498db")
    ax2.plot(df.index, df[col2], linewidth=1,   label=col2.replace("fund_",""), color="#e67e22", linestyle="--")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper left")
    ax.set_ylabel(title, fontsize=8)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
fig.autofmt_xdate()
save(fig, "10_fundamentals")


# 11. CORRELATION WITH TARGET
print("\n[11] Correlation with target")

target_col = "target_1d"
numeric_df  = df.select_dtypes(include="number").dropna(subset=[target_col])
corr        = numeric_df.corr()[target_col].drop(target_col).sort_values()

print("  Top 10 positive correlates:")
print(corr.tail(10).to_string())
print("  Top 10 negative correlates:")
print(corr.head(10).to_string())

# Bar chart — top/bottom 15 each
n = 15
top_corr = pd.concat([corr.head(n), corr.tail(n)]).sort_values()
colors   = ["#e74c3c" if v < 0 else "#2ecc71" for v in top_corr]

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(top_corr.index, top_corr.values, color=colors, edgecolor="none")
ax.axvline(0, color="white", linewidth=0.8)
ax.set_xlabel(f"Pearson correlation with {target_col}")
ax.set_title(f"{SYMBOL} — Top {n} +/− Feature Correlations with Next-Day Return",
             fontweight="bold")
fig.tight_layout()
save(fig, "11_correlation_target")

# Full heatmap for key feature groups
key_cols = [
    "daily_return","log_return","RSI_14","MACD","bb_pct",
    "realized_vol_20d","vol_of_vol","momentum_1m","momentum_3m",
    "momentum_6m","momentum_12m","vix_level","yield_10y",
    "spy_beta_60d","news_sent_score","news_sent_marketaux",
    "fund_gross_margin","fund_roe","opt_atm_iv","target_1d"
]
key_cols = [c for c in key_cols if c in df.columns]
corr_mat = numeric_df[key_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
sns.heatmap(corr_mat, ax=ax, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.3,
            annot_kws={"size": 7}, vmin=-1, vmax=1)
ax.set_title(f"{SYMBOL} — Feature Correlation Heatmap (key groups)", fontweight="bold")
fig.tight_layout()
save(fig, "11b_correlation_heatmap")


# 12. FEATURE DISTRIBUTIONS
print("\n[12] Feature distributions")

num_cols = df.select_dtypes(include="number").columns.tolist()
num_cols = [c for c in num_cols if c != "target_1d"]

n_cols = 4
n_rows = int(np.ceil(len(num_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
fig.suptitle(f"{SYMBOL} — Feature Distributions", fontweight="bold", y=1.01)
axes_flat = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes_flat[i]
    data = df[col].dropna()
    ax.hist(data, bins=40, edgecolor="none", alpha=0.8)
    ax.set_title(col, fontsize=7, pad=2)
    ax.tick_params(labelsize=6)
    skew = data.skew()
    ax.annotate(f"skew={skew:.2f}", xy=(0.97, 0.92), xycoords="axes fraction",
                ha="right", fontsize=6, color="white")

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.tight_layout()
save(fig, "12_feature_distributions")


print(f"\n=== EDA complete — {len(list(OUT_DIR.glob('*.png')))} plots saved to {OUT_DIR} ===")
