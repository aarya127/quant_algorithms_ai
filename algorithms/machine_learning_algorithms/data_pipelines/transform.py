"""
transform.py — Unified feature-matrix builder for ML pipelines.

Merges every data source from DataExtractor into a single daily-indexed
DataFrame where each row is one trading day and each column is a named
feature ready for model consumption.

Merge strategy
--------------
  Spine          : technicals()               — OHLCV + computed indicators (daily)
  Derived        : computed in-place          — %B, price/SMA ratios, volume z-score
  Point-in-time  : historical_fundamentals()  — quarterly financials via merge_asof
  Aggregated     : news()                     — daily article count
  Resampled      : sentiment()                — monthly insider MSPR forward-filled
  Snapshot       : options()                  — ATM-IV, P/C-OI ratio, max-pain
  Event flags    : earnings_calendar()        — proximity & binary earnings flags

Point-in-time note
------------------
Fundamental columns are computed from yfinance quarterly statements (income,
balance sheet, cash flow) and joined via pd.merge_asof(direction="backward") so
every trading day sees only the most recently filed quarterly values — no
look-ahead bias for time-series model training.  The options snapshot is still
today's surface and should be treated as a live-only feature.

Quick start
-----------
    from algorithms.machine_learning_algorithms.data_pipelines.transform import DataTransformer

    dt  = DataTransformer()
    df  = dt.build_feature_matrix("NVDA", period="1y")
    print(df.shape)                # (trading_days, ~65 features)
    dt.describe_columns(df)        # prints column catalogue with null %
"""

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from .extractor import DataExtractor

logger = logging.getLogger("DataTransformer")

# ---------------------------------------------------------------------------
# Column catalogue (used by describe_columns)
# ---------------------------------------------------------------------------
_COL_DESC: dict = {
    # OHLCV
    "symbol":        "Ticker symbol",
    "Open":          "Daily open price (adjusted)",
    "High":          "Daily high price (adjusted)",
    "Low":           "Daily low price (adjusted)",
    "Close":         "Daily close price (adjusted)",
    "Volume":        "Daily traded volume",
    # Technicals
    "SMA_20":        "20-day simple moving average of close",
    "SMA_50":        "50-day simple moving average of close",
    "SMA_200":       "200-day simple moving average of close",
    "EMA_20":        "20-day exponential moving average of close",
    "RSI_14":        "Relative Strength Index (14-period); 30/70 oversold/overbought",
    "MACD":          "MACD line = EMA12 − EMA26",
    "MACD_signal":   "MACD signal line = EMA9 of MACD",
    "MACD_hist":     "MACD histogram = MACD − signal (momentum direction)",
    "BB_upper":      "Bollinger upper band = SMA20 + 2σ",
    "BB_mid":        "Bollinger middle band = SMA20",
    "BB_lower":      "Bollinger lower band = SMA20 − 2σ",
    "ATR_14":        "Average True Range (14-period); proxy for daily volatility",
    "OBV":           "On-Balance Volume (cumulative volume direction)",
    "daily_return":  "Simple daily return (Close_t / Close_{t-1} − 1)",
    "log_return":    "Log return ln(Close_t / Close_{t-1})",
    # Derived price ratios
    "price_to_sma20":  "Close / SMA_20 — mean-reversion signal (1.0 = at MA)",
    "price_to_sma50":  "Close / SMA_50",
    "price_to_sma200": "Close / SMA_200 — trend regime (>1 = above LT average)",
    "bb_pct":          "Bollinger %B = (Close − BB_lower) / (BB_upper − BB_lower); 0–1 band position",
    "volume_zscore":   "Volume z-score over rolling 20-day window (spikes = unusual activity)",
    "high_low_range":  "Intraday range = (High − Low) / Close",
    "overnight_gap":   "Overnight gap = Open_t / Close_{t-1} − 1",
    # Volatility & risk
    "realized_vol_20d":   "Annualised 20-day realised volatility (rolling std of log_return × √252)",
    "realized_vol_60d":   "Annualised 60-day realised volatility",
    "vol_of_vol":         "Rolling 20-day std of realized_vol_20d (vol-of-vol / vol regime indicator)",
    "skewness_20d":       "20-day rolling skewness of log returns (negative = fat left tail)",
    "kurtosis_20d":       "20-day rolling excess kurtosis of log returns (>3 = fat tails)",
    "downside_deviation": "20-day annualised semi-deviation using only negative log returns",
    "max_drawdown_20d":   "Max peak-to-trough drawdown over trailing 20 trading days (≤0)",
    # Microstructure
    "amihud_illiquidity":    "Amihud (2002): 20-day mean of |return|/volume×1e6 — higher = less liquid",
    "bid_ask_spread_proxy":  "Corwin-Schultz spread proxy = (High−Low)/mid; higher = wider spread",
    "volume_participation":  "Today's volume / 20-day average volume (>1 = above-average activity)",
    "trade_imbalance":       "(Close−Open)/(High−Low); +1 = closed at high, −1 = closed at low",
    # Momentum & mean-reversion
    "reversal_1w":   "5-day simple return (short-term reversal factor)",
    "momentum_1m":   "21-day return skipping last 5 days (Jegadeesh-Titman 1-month)",
    "momentum_3m":   "63-day return skipping last 5 days (Jegadeesh-Titman 3-month)",
    "momentum_6m":   "126-day return skipping last 5 days (Jegadeesh-Titman 6-month)",
    "momentum_12m":  "252-day return skipping last 5 days (Jegadeesh-Titman 12-month)",
    "high_52w_pct":  "Distance from 52-week high = Close/max252 − 1 (≤0; 0 = at high)",
    "low_52w_pct":   "Distance from 52-week low = Close/min252 − 1 (≥0; 0 = at low)",
    "price_accel":   "Momentum acceleration = momentum_1m − momentum_3m",
    # Cross-asset / macro
    "vix_level":          "CBOE VIX daily close (fear gauge; >30 = elevated uncertainty)",
    "yield_10y":          "US 10-year Treasury yield (%) from ^TNX",
    "yield_curve_slope":  "10Y yield minus 3M yield (bps proxy); negative = inverted curve",
    "spy_beta_60d":       "Rolling 60-day OLS beta vs SPY (systematic market sensitivity)",
    "qqq_corr_20d":       "Rolling 20-day Pearson correlation of log returns vs QQQ",
    "sox_rel_strength":   "Daily log return minus SOXX (semiconductor ETF) log return",
    # Fundamentals (point-in-time quarterly — merged via merge_asof, no look-ahead)
    "fund_rev_ttm":          "TTM total revenue (sum of 4 most recent quarters)",
    "fund_gross_profit_ttm": "TTM gross profit",
    "fund_net_income_ttm":   "TTM net income",
    "fund_eps_ttm":          "TTM diluted EPS",
    "fund_fcf_ttm":          "TTM free cash flow (operating CF + capex)",
    "fund_gross_margin":     "Gross margin = gross_profit_ttm / rev_ttm",
    "fund_operating_margin": "Operating margin = op_income_ttm / rev_ttm",
    "fund_net_margin":       "Net margin = net_income_ttm / rev_ttm",
    "fund_roe":              "Return on equity = net_income_ttm / stockholders_equity",
    "fund_roa":              "Return on assets = net_income_ttm / total_assets",
    "fund_debt_to_equity":   "Debt-to-equity = total_debt / stockholders_equity",
    "fund_current_ratio":    "Current ratio = current_assets / current_liabilities",
    "fund_total_assets":     "Point-in-time total assets (balance sheet snapshot)",
    "fund_total_debt":       "Point-in-time total debt (balance sheet snapshot)",
    "fund_cash":             "Point-in-time cash and equivalents",
    "fund_rev_growth_yoy":   "YoY TTM revenue growth (pct change vs 4 quarters prior)",
    "fund_quarter_end":      "Quarter-end date this row's fundamentals were reported",
    # News
    "news_count":       "Number of news articles indexed on this trading day",
    "news_sent_score":  "Daily mean sentiment score from Polygon/AV/Finnhub (-1=bearish, 0=neutral, +1=bullish)",
    "news_sent_7d":     "7-trading-day rolling mean of news_sent_score (smoothed signal)",
    # Insider sentiment (monthly, forward-filled to daily)
    "insdr_change":   "Net insider shares (bought − sold) for the month; positive = net buying",
    "insdr_mspr":     "Buy ratio = bought_shares / (bought + sold); 0.5 = neutral, >0.5 = net buying",
    # Options snapshot (current chain, static broadcast)
    "opt_atm_iv":     "Implied volatility of the ATM call (nearest strike to last close)",
    "opt_put_call_oi":"Put OI / Call OI for front expiry (>1 = bearish positioning)",
    "opt_total_oi":   "Total open interest (calls + puts) across front expiry",
    "opt_max_pain":   "Max-pain strike: OI-weighted strike causing maximum option-buyer loss",
    "opt_as_of":      "Date the options snapshot was taken",
    # Earnings
    "earn_days_to_next":    "Calendar days until next earnings announcement",
    "earn_days_since_last": "Calendar days since last earnings announcement",
    "earn_is_week":         "1 if within ±5 calendar days of an earnings event, else 0",
}


# ---------------------------------------------------------------------------
# DataTransformer
# ---------------------------------------------------------------------------

class DataTransformer:
    """
    Builds a unified, daily-indexed ML feature matrix by merging all
    DataExtractor sources for a given symbol.

    Parameters
    ----------
    extractor : DataExtractor | None
        Provide an existing instance to reuse authenticated SDK clients;
        a fresh DataExtractor is created when None (default).
    """

    def __init__(self, extractor: Optional[DataExtractor] = None):
        self.ex = extractor or DataExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self,
        symbol:   str,
        start:    str = "",
        end:      str = "",
        period:   str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Build and return a fully merged daily feature matrix for *symbol*.

        Parameters
        ----------
        symbol   : ticker, e.g. 'NVDA'
        start    : 'YYYY-MM-DD' — if both start+end provided, used instead of period
        end      : 'YYYY-MM-DD'
        period   : yfinance period string ('1d','5d','1mo','3mo','6mo',
                   '1y','2y','5y','10y','ytd','max')
        interval : bar size (only '1d' recommended for feature matrices)

        Returns
        -------
        pd.DataFrame
            Index  : Date (tz-naive daily)
            Columns: symbol + ~65 named feature columns
        """
        sym = symbol.upper()
        logger.info("Building feature matrix for %s (period=%s)", sym, period)

        # ── 1. Spine: OHLCV + computed technical indicators ─────────────
        # Longest rolling window is SMA_200 (200 trading days ≈ 285 calendar
        # days). Fetch a warmup buffer before the requested start so that the
        # first bar in the output period has fully-warmed indicators instead
        # of NaN. We trim the warmup rows back out before returning.
        _WARMUP_CALENDAR_DAYS = 400   # covers SMA_200 + momentum_12m (shift(257) ≈ 372 cal days)

        # Map period strings to approximate calendar-day counts
        _PERIOD_DAYS: dict = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 91, "6mo": 182,
            "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
        }

        today = datetime.date.today()

        if start and end:
            # Explicit date range supplied — extend start backward for warmup
            requested_start = datetime.date.fromisoformat(start)
            warmup_start    = requested_start - datetime.timedelta(days=_WARMUP_CALENDAR_DAYS)
            spine = self.ex.technicals(sym, start=warmup_start.isoformat(),
                                       end=end, interval=interval)
        elif period in _PERIOD_DAYS:
            requested_start = today - datetime.timedelta(days=_PERIOD_DAYS[period])
            warmup_start    = requested_start - datetime.timedelta(days=_WARMUP_CALENDAR_DAYS)
            spine = self.ex.technicals(sym, start=warmup_start.isoformat(),
                                       end=today.isoformat(), interval=interval)
        else:
            # Unknown period (e.g. 'ytd', 'max') — no warmup logic, use as-is
            requested_start = None
            spine = self.ex.technicals(sym, period=period, interval=interval)

        if spine is None or spine.empty:
            logger.error("No OHLCV data for %s — aborting", sym)
            return pd.DataFrame()

        # Normalise index to tz-naive for consistent joining across sources
        spine.index = pd.to_datetime(spine.index).tz_localize(None)
        spine.index.name = "Date"
        spine["symbol"] = sym

        # ── Trim warmup rows: keep only the originally-requested date range ──
        if requested_start is not None:
            cutoff = pd.Timestamp(requested_start)
            spine = spine[spine.index >= cutoff]

        # Determine date bounds from spine (used for ranged sub-queries)
        start_str = spine.index.min().date().isoformat()
        end_str   = spine.index.max().date().isoformat()
        logger.info("[1/7] Spine ready: %d trading days  %s → %s", len(spine), start_str, end_str)

        # ── 2. Derived price / volatility ratios ────────────────────────
        spine = self._add_derived_features(spine)
        logger.info("[2/11] Derived features added: %d cols", len(spine.columns))

        # ── 3. Risk / volatility features ────────────────────────────────
        spine = self._add_risk_features(spine)
        risk_cols = [c for c in spine.columns if c in (
            "realized_vol_20d","realized_vol_60d","vol_of_vol",
            "skewness_20d","kurtosis_20d","downside_deviation","max_drawdown_20d")]
        logger.info("[3/11] Risk features added: %d cols", len(risk_cols))

        # ── 4. Microstructure features ────────────────────────────────────
        spine = self._add_microstructure_features(spine)
        micro_cols = [c for c in spine.columns if c in (
            "amihud_illiquidity","bid_ask_spread_proxy",
            "volume_participation","trade_imbalance")]
        logger.info("[4/11] Microstructure features added: %d cols", len(micro_cols))

        # ── 5. Momentum / mean-reversion features ────────────────────────
        spine = self._add_momentum_features(spine)
        mom_cols = [c for c in spine.columns if c in (
            "momentum_1m","momentum_3m","momentum_6m","momentum_12m",
            "reversal_1w","high_52w_pct","low_52w_pct","price_accel")]
        logger.info("[5/11] Momentum features added: %d cols", len(mom_cols))

        # ── 6. Cross-asset / macro features ──────────────────────────────
        spine = self._merge_macro_features(spine, start_str, end_str)
        macro_cols = [c for c in spine.columns if c in (
            "vix_level","yield_10y","yield_curve_slope",
            "spy_beta_60d","qqq_corr_20d","sox_rel_strength")]
        logger.info("[6/11] Macro/cross-asset features added: %d cols", len(macro_cols))

        # ── 7. Fundamentals (broadcast across all dates) ─────────────────
        spine = self._merge_fundamentals(spine, sym)
        fund_cols = [c for c in spine.columns if c.startswith('fund_')]
        logger.info("[7/11] Fundamentals merged: %d fund_ cols, first valid: %s",
                    len(fund_cols),
                    spine[fund_cols[0]].first_valid_index() if fund_cols else 'N/A')

        # ── 8. News — daily article count + sentiment ────────────────────
        spine = self._merge_news(spine, sym, start_str, end_str)
        for _nc in ('news_sent_av', 'news_sent_polygon', 'news_sent_finnhub', 'news_sent_score'):
            if _nc in spine.columns:
                _n = spine[_nc].notna().sum()
                logger.info("[8/11] %-24s: %d/%d rows filled (%.0f%%)",
                            _nc, _n, len(spine), _n / len(spine) * 100)

        # ── 9. Insider sentiment (monthly → daily by forward-fill) ───────
        spine = self._merge_sentiment(spine, sym)
        insdr_cols = [c for c in spine.columns if c.startswith('insdr_')]
        logger.info("[9/11] Insider sentiment: %d insdr_ cols", len(insdr_cols))

        # ── 10. Options snapshot — surface metrics ────────────────────────
        spine = self._merge_options_snapshot(spine, sym)
        opt_cols = [c for c in spine.columns if c.startswith('opt_')]
        logger.info("[10/11] Options snapshot: %d opt_ cols", len(opt_cols))

        # ── 11. Earnings calendar proximity flags ─────────────────────────
        spine = self._merge_earnings(spine, sym)
        earn_cols = [c for c in spine.columns if c.startswith('earn_')]
        logger.info("[11/11] Earnings flags: %d earn_ cols", len(earn_cols))

        # ── 12. Column ordering: symbol first, then chronological groups ──
        group_order = (
            ["symbol"]
            + [c for c in spine.columns if c in ("Open","High","Low","Close","Volume")]
            + [c for c in spine.columns if c in ("SMA_20","SMA_50","SMA_200","EMA_20",
               "RSI_14","MACD","MACD_signal","MACD_hist","BB_upper","BB_mid","BB_lower",
               "ATR_14","OBV","daily_return","log_return")]
            + [c for c in spine.columns if c.startswith("price_to_") or c in
               ("bb_pct","volume_zscore","high_low_range","overnight_gap")]
            # Risk / volatility
            + [c for c in spine.columns if c in (
               "realized_vol_20d","realized_vol_60d","vol_of_vol",
               "skewness_20d","kurtosis_20d","downside_deviation","max_drawdown_20d")]
            # Microstructure
            + [c for c in spine.columns if c in (
               "amihud_illiquidity","bid_ask_spread_proxy",
               "volume_participation","trade_imbalance")]
            # Momentum
            + [c for c in spine.columns if c in (
               "reversal_1w","momentum_1m","momentum_3m","momentum_6m","momentum_12m",
               "high_52w_pct","low_52w_pct","price_accel")]
            # Cross-asset / macro
            + [c for c in spine.columns if c in (
               "vix_level","yield_10y","yield_curve_slope",
               "spy_beta_60d","qqq_corr_20d","sox_rel_strength")]
            + sorted([c for c in spine.columns if c.startswith("fund_")])
            + [c for c in spine.columns if c.startswith("news_")]
            + [c for c in spine.columns if c.startswith("insdr_")]
            + [c for c in spine.columns if c.startswith("opt_")]
            + [c for c in spine.columns if c.startswith("earn_")]
            # anything else (Dividends, Stock Splits, etc.)
            + [c for c in spine.columns if c not in set(spine.columns[:0])]
        )
        # Deduplicate while preserving order
        seen, ordered = set(), []
        for c in group_order:
            if c in spine.columns and c not in seen:
                seen.add(c)
                ordered.append(c)

        spine = spine[ordered]
        logger.info("Feature matrix ready: %d rows × %d columns", *spine.shape)
        return spine

    @staticmethod
    def describe_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Print a formatted catalogue of every column in *df* with its dtype,
        null-percentage, and human-readable description.

        Returns the catalogue as a DataFrame for programmatic use.
        """
        rows = []
        for col in df.columns:
            null_pct = round(df[col].isna().mean() * 100, 1)
            rows.append({
                "column":      col,
                "dtype":       str(df[col].dtype),
                "null_%":      null_pct,
                "description": _COL_DESC.get(col, "—"),
            })
        desc = pd.DataFrame(rows)
        print(desc.to_string(index=False))
        return desc

    # ------------------------------------------------------------------
    # Private merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute ratio / normalisation features entirely from spine columns."""
        c    = df["Close"]
        high = df["High"]
        low  = df["Low"]

        if "SMA_20" in df.columns:
            df["price_to_sma20"]  = (c / df["SMA_20"]).replace([np.inf, -np.inf], np.nan)
        if "SMA_50" in df.columns:
            df["price_to_sma50"]  = (c / df["SMA_50"]).replace([np.inf, -np.inf], np.nan)
        if "SMA_200" in df.columns:
            df["price_to_sma200"] = (c / df["SMA_200"]).replace([np.inf, -np.inf], np.nan)

        if "BB_upper" in df.columns and "BB_lower" in df.columns:
            band = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
            df["bb_pct"] = (c - df["BB_lower"]) / band

        if "Volume" in df.columns:
            vol = df["Volume"].astype(float)
            roll_mean = vol.rolling(20, min_periods=5).mean()
            roll_std  = vol.rolling(20, min_periods=5).std().replace(0, np.nan)
            df["volume_zscore"] = (vol - roll_mean) / roll_std

        df["high_low_range"] = ((high - low) / c.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        df["overnight_gap"] = (df["Open"] / c.shift(1).replace(0, np.nan) - 1).replace(
            [np.inf, -np.inf], np.nan
        )
        return df

    def _merge_fundamentals(self, df: pd.DataFrame, sym: str) -> pd.DataFrame:
        """
        Point-in-time merge of quarterly fundamentals onto the daily spine.

        Uses pd.merge_asof(direction='backward') so each trading day only sees
        the most recently filed quarterly report — zero look-ahead bias.
        """
        try:
            fund = self.ex.historical_fundamentals(sym)
            if fund is None or fund.empty:
                return df

            # merge_asof requires both keys sorted ascending
            spine_reset = df.reset_index()          # Date column
            fund_reset  = fund.reset_index()        # Date column

            merged = pd.merge_asof(
                spine_reset.sort_values("Date"),
                fund_reset.sort_values("Date"),
                on="Date",
                direction="backward",
            )
            merged = merged.set_index("Date")
            # Restore original row order
            df = merged.reindex(df.index)

        except Exception as exc:
            logger.warning("Fundamentals merge failed for %s: %s", sym, exc)
        return df

    def _merge_news(
        self, df: pd.DataFrame, sym: str, start: str, end: str
    ) -> pd.DataFrame:
        """Merge daily article count + sentiment scores into the spine.

        Adds eight columns:
            news_count            — articles published on each trading day (all sources)
            news_sent_score       — blended mean sentiment score (-1 to +1)
            news_sent_av          — Alpha Vantage daily weighted-mean score (-1 to +1)
            news_sent_polygon     — Polygon / Alpaca AI-insight daily score (-1 to +1)
            news_sent_finnhub     — Finnhub headlines scored by FinBERT (-1 to +1)
            news_sent_7d          — 7-trading-day rolling mean of blended score
            news_sent_av_7d       — 7-trading-day rolling mean of AV score
            news_sent_polygon_7d  — 7-trading-day rolling mean of Polygon score
            news_sent_finnhub_7d  — 7-trading-day rolling mean of Finnhub FinBERT score

        Source priority (handled inside extractor):
            1. Polygon /v3/reference/news  (AI insights per ticker)
            2. Alpha Vantage NEWS_SENTIMENT (relevance-weighted scores)
            3. Finnhub company news         (count only, score = NaN)
        """
        try:
            daily = self.ex.news_sentiment_history(sym, start=start, end=end)

            if daily is not None and not daily.empty:
                daily.index = pd.to_datetime(daily.index).tz_localize(None)
                logger.info("  news_sentiment_history returned %d article-days  %s → %s",
                            len(daily), daily.index.min().date(), daily.index.max().date())
                for _src_col in ('news_sent_av', 'news_sent_polygon', 'news_sent_finnhub'):
                    if _src_col in daily.columns:
                        _n = daily[_src_col].notna().sum()
                        logger.info("  %-24s: %d/%d article-days with scores", _src_col, _n, len(daily))

                # Articles published on weekends/holidays need to roll forward to
                # the next trading day. Use merge_asof(direction="nearest") so each
                # calendar-date entry attaches to the nearest trading day.
                spine_dates = df.index.to_frame(index=False).rename(columns={"Date": "trade_date"})
                daily_reset = daily.reset_index().rename(columns={"date": "trade_date"})
                daily_reset["trade_date"] = daily_reset["trade_date"].astype("datetime64[us]")
                spine_dates["trade_date"] = spine_dates["trade_date"].astype("datetime64[us]")

                merged = pd.merge_asof(
                    spine_dates.sort_values("trade_date"),
                    daily_reset.sort_values("trade_date"),
                    on="trade_date",
                    direction="nearest",   # snap weekend articles to nearest trading day
                    tolerance=pd.Timedelta("4d"),  # max 4 calendar days gap
                ).set_index("trade_date")

                df["news_sent_score"]   = merged["news_sent_score"].values
                df["news_sent_av"]      = merged["news_sent_av"].values if "news_sent_av" in merged.columns else float("nan")
                df["news_sent_polygon"] = merged["news_sent_polygon"].values if "news_sent_polygon" in merged.columns else float("nan")
                df["news_sent_finnhub"] = merged["news_sent_finnhub"].values if "news_sent_finnhub" in merged.columns else float("nan")
                df["news_count"]        = merged["news_articles"].fillna(0).astype(int).values

                logger.info("  After merge_asof (tol=4d): news_sent_av filled %d/%d trading days",
                            df["news_sent_av"].notna().sum(), len(df))

                # Forward-fill sentiment scores: carry the last known score into
                # days with no articles (no look-ahead bias — scores are based
                # on already-published articles).  Cap at 20 trading days (~4 weeks)
                # so a stale signal doesn't propagate across an extended quiet spell.
                # Back-fill covers the very start of the window where no prior
                # article may exist yet (limited to 5 days to avoid large back-smear).
                _FFILL_LIMIT = 20  # trading days
                _BFILL_LIMIT = 5   # trading days (start-of-window only)
                for _sc in ("news_sent_score", "news_sent_av", "news_sent_polygon", "news_sent_finnhub"):
                    df[_sc] = df[_sc].ffill(limit=_FFILL_LIMIT).bfill(limit=_BFILL_LIMIT)

                logger.info("  After ffill(%d)/bfill(%d): news_sent_av filled %d/%d trading days",
                            _FFILL_LIMIT, _BFILL_LIMIT, df["news_sent_av"].notna().sum(), len(df))

                # 7-day rolling means for each score column
                df["news_sent_7d"] = (
                    df["news_sent_score"].rolling(7, min_periods=1).mean()
                )
                df["news_sent_av_7d"] = (
                    df["news_sent_av"].rolling(7, min_periods=1).mean()
                )
                df["news_sent_polygon_7d"] = (
                    df["news_sent_polygon"].rolling(7, min_periods=1).mean()
                )
                df["news_sent_finnhub_7d"] = (
                    df["news_sent_finnhub"].rolling(7, min_periods=1).mean()
                )
            else:
                # Fallback: raw article count from the simple news() method
                news = self.ex.news(sym, start=start, end=end, limit=500)
                if news is not None and not news.empty:
                    dates = (
                        pd.to_datetime(news["datetime"])
                        .dt.tz_localize(None)
                        .dt.normalize()
                    )
                    daily_ct = (
                        dates.value_counts()
                        .rename_axis("Date")
                        .rename("news_count")
                        .sort_index()
                    )
                    df = df.join(daily_ct, how="left")
                df["news_count"]        = df.get("news_count", 0).fillna(0).astype(int)
                df["news_sent_score"]   = float("nan")
                df["news_sent_av"]      = float("nan")
                df["news_sent_polygon"] = float("nan")
                df["news_sent_finnhub"] = float("nan")
                df["news_sent_7d"]         = float("nan")
                df["news_sent_av_7d"]      = float("nan")
                df["news_sent_polygon_7d"] = float("nan")
                df["news_sent_finnhub_7d"] = float("nan")

        except Exception as exc:
            logger.warning("News merge failed for %s: %s", sym, exc)
            df["news_count"]        = 0
            df["news_sent_score"]   = float("nan")
            df["news_sent_av"]      = float("nan")
            df["news_sent_polygon"] = float("nan")
            df["news_sent_finnhub"] = float("nan")
            df["news_sent_7d"]         = float("nan")
            df["news_sent_av_7d"]      = float("nan")
            df["news_sent_polygon_7d"] = float("nan")
            df["news_sent_finnhub_7d"] = float("nan")
        return df

    def _merge_sentiment(self, df: pd.DataFrame, sym: str) -> pd.DataFrame:
        """Forward-fill monthly insider MSPR into a daily column."""
        try:
            sent = self.ex.sentiment(sym)
            if sent is None or sent.empty:
                return df

            # Build month-start DatetimeIndex from year + month columns
            monthly = sent.copy()
            monthly["Date"] = pd.to_datetime(
                monthly["year"].astype(str) + "-"
                + monthly["month"].astype(str).str.zfill(2)
            )
            monthly = (
                monthly[["Date", "change", "mspr"]]
                .rename(columns={"change": "insdr_change", "mspr": "insdr_mspr"})
                .sort_values("Date")
            )
            # Align datetime precision to the trading spine (yfinance → datetime64[us])
            monthly["Date"] = monthly["Date"].astype("datetime64[us]")

            # merge_asof (backward) handles month-starts that fall on weekends/holidays:
            # each trading day gets the most recently published monthly figure.
            # Strip timezone (yfinance returns tz-aware index) and normalize to us precision
            raw_idx = df.index.tz_localize(None) if df.index.tz is None else df.index.tz_convert(None)
            spine = pd.DataFrame({"Date": raw_idx.astype("datetime64[us]")})
            combined = pd.merge_asof(
                spine,
                monthly,
                on="Date",
                direction="backward",
            ).set_index("Date")
            df["insdr_change"] = combined["insdr_change"].values
            df["insdr_mspr"]   = combined["insdr_mspr"].values

        except Exception as exc:
            logger.warning("Sentiment merge failed for %s: %s", sym, exc)
        return df

    def _merge_options_snapshot(self, df: pd.DataFrame, sym: str) -> pd.DataFrame:
        """
        Derive four options-surface scalars from the nearest liquid expiry and
        broadcast them as constant columns across the spine.

        Expiry selection: the first expiry with total OI > 0; falls back to
        the second-nearest expiry when the front expiry has zero OI (e.g.
        same-day / next-day expiries that have already settled).

        Metrics
        -------
        opt_atm_iv      — IV of the call with strike nearest to last close
        opt_put_call_oi — put OI / call OI (bearish positioning > 1)
        opt_total_oi    — total OI across all strikes in chosen expiry
        opt_max_pain    — strike that minimises total OI-weighted intrinsic payout
                          to option buyers (widely watched by MM desks)
        """
        try:
            opts = self.ex.options(sym, contract_type="both", enrich_polygon=False)
            if opts is None or opts.empty:
                return df

            last_close = float(df["Close"].iloc[-1])

            # Pick the expiry with the most total OI — skips zero-OI same-day expiries
            if "expiration" in opts.columns and "openInterest" in opts.columns:
                oi_by_exp = (
                    opts.groupby("expiration")["openInterest"]
                    .sum()
                    .sort_index()
                )
                liquid = oi_by_exp[oi_by_exp > 0]
                front_exp = liquid.index[0] if not liquid.empty else oi_by_exp.index[0]
            else:
                front_exp = opts["expiration"].min() if "expiration" in opts.columns else None

            atm_iv = put_call_oi = total_oi = max_pain = float("nan")

            if front_exp:
                front  = opts[opts["expiration"] == front_exp]
                calls  = front[front["optionType"] == "call"].copy()
                puts   = front[front["optionType"] == "put"].copy()

                # ATM IV — nearest call with a meaningful IV (>= 15% threshold)
                # yfinance returns degenerate IV (exact binary fractions: 1/16, 1/8…)
                # when bid=ask=0 (after-hours / market-closed). No individual stock
                # option has genuine IV below 15%, so we discard those rows.
                _MIN_IV = 0.15   # 15% — below this the IV is synthesis artefact
                if not calls.empty and "strike" in calls.columns and "impliedVolatility" in calls.columns:
                    valid_calls = calls[calls["impliedVolatility"] >= _MIN_IV]
                    if not valid_calls.empty:
                        nearest_idx = (valid_calls["strike"] - last_close).abs().idxmin()
                        atm_iv = float(valid_calls.loc[nearest_idx, "impliedVolatility"])
                    # else: atm_iv stays NaN — data unavailable

                # Put / call OI ratio — NaN when total OI is 0 (not yet settled)
                call_oi = float(calls["openInterest"].fillna(0).sum()) if "openInterest" in calls.columns else 0
                put_oi  = float(puts["openInterest"].fillna(0).sum())  if "openInterest" in puts.columns  else 0
                total_oi    = int(call_oi + put_oi)
                put_call_oi = round(put_oi / call_oi, 4) if call_oi > 0 else float("nan")
                if total_oi == 0:
                    total_oi = float("nan")  # distinguish "no data" from genuine 0
                    atm_iv   = float("nan")  # OI=0 means data is stale/unsettled

                # Max-pain: only meaningful when OI > 0
                if "openInterest" in front.columns and total_oi > 0:
                    strikes = sorted(front["strike"].unique())
                    pain    = {}
                    c_oi = calls.set_index("strike")["openInterest"].fillna(0)
                    p_oi = puts.set_index("strike")["openInterest"].fillna(0)
                    for s in strikes:
                        # Call holders lose when strike > s (their calls expire worthless)
                        c_loss = sum(max(s - k, 0) * oi for k, oi in c_oi.items())
                        # Put holders lose when strike < s
                        p_loss = sum(max(k - s, 0) * oi for k, oi in p_oi.items())
                        pain[s] = c_loss + p_loss
                    max_pain = float(min(pain, key=pain.get))

            df["opt_atm_iv"]      = atm_iv
            df["opt_put_call_oi"] = put_call_oi
            df["opt_total_oi"]    = total_oi
            df["opt_max_pain"]    = max_pain
            df["opt_as_of"]       = datetime.date.today().isoformat()

        except Exception as exc:
            logger.warning("Options snapshot merge failed for %s: %s", sym, exc)
        return df

    # ------------------------------------------------------------------
    # New feature groups
    # ------------------------------------------------------------------

    @staticmethod
    def _add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility and downside-risk features computed entirely from log_return.

        Adds:
            realized_vol_20d    — annualised 20-day realised volatility
            realized_vol_60d    — annualised 60-day realised volatility
            vol_of_vol          — rolling 20-day std of realized_vol_20d (vol regime)
            skewness_20d        — 20-day rolling skewness of log returns
            kurtosis_20d        — 20-day rolling excess kurtosis of log returns
            downside_deviation  — 20-day semi-deviation (std of negative returns only)
            max_drawdown_20d    — max peak-to-trough drawdown over rolling 20-day window
        """
        if "log_return" not in df.columns:
            return df
        lr = df["log_return"]

        df["realized_vol_20d"] = lr.rolling(20, min_periods=5).std() * np.sqrt(252)
        df["realized_vol_60d"] = lr.rolling(60, min_periods=20).std() * np.sqrt(252)
        df["vol_of_vol"] = df["realized_vol_20d"].rolling(20, min_periods=5).std()

        df["skewness_20d"] = lr.rolling(20, min_periods=10).skew()
        df["kurtosis_20d"] = lr.rolling(20, min_periods=10).kurt()

        # Downside deviation: std of returns below zero only
        def _downside_std(x: pd.Series) -> float:
            neg = x[x < 0]
            return neg.std() * np.sqrt(252) if len(neg) >= 3 else np.nan
        df["downside_deviation"] = lr.rolling(20, min_periods=10).apply(
            _downside_std, raw=False
        )

        # Max drawdown over rolling 20-day window using Close prices
        if "Close" in df.columns:
            close = df["Close"]
            roll_max = close.rolling(20, min_periods=1).max()
            df["max_drawdown_20d"] = (close - roll_max) / roll_max.replace(0, np.nan)

        return df

    @staticmethod
    def _add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Market microstructure proxies computed from daily OHLCV.

        Adds:
            amihud_illiquidity      — 20-day mean of |return|/volume (×1e6 for scale)
            bid_ask_spread_proxy    — (High−Low)/mid-price (Corwin-Schultz proxy)
            volume_participation    — today's volume / 20-day average volume
            trade_imbalance         — (Close−Open)/(High−Low); direction of intraday flow
        """
        close  = df.get("Close")
        high   = df.get("High")
        low    = df.get("Low")
        volume = df.get("Volume")
        open_  = df.get("Open")
        lr     = df.get("log_return")

        if close is None or volume is None or lr is None:
            return df

        # Amihud illiquidity — scale by 1e6 so values are in readable range
        vol_f = volume.astype(float).replace(0, np.nan)
        amihud_daily = lr.abs() / vol_f * 1e6
        df["amihud_illiquidity"] = amihud_daily.rolling(20, min_periods=5).mean()

        # Bid-ask spread proxy
        if high is not None and low is not None:
            mid = (high + low) / 2
            df["bid_ask_spread_proxy"] = ((high - low) / mid.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            )

        # Volume participation rate
        avg_vol = vol_f.rolling(20, min_periods=5).mean()
        df["volume_participation"] = (vol_f / avg_vol.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )

        # Intraday trade imbalance: +1 = closed at high, −1 = closed at low
        if high is not None and low is not None and open_ is not None:
            hl_range = (high - low).replace(0, np.nan)
            df["trade_imbalance"] = ((close - open_) / hl_range).clip(-1, 1)

        return df

    @staticmethod
    def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional momentum and mean-reversion features from Close prices.

        Adds:
            momentum_1m     — 21-day return (skip last 5d)
            momentum_3m     — 63-day return (skip last 5d)
            momentum_6m     — 126-day return (skip last 5d)
            momentum_12m    — 252-day return (skip last 5d) [Jegadeesh-Titman]
            reversal_1w     — last 5-day return (short-term reversal factor)
            high_52w_pct    — distance from 52-week high (0 = AT high)
            low_52w_pct     — distance from 52-week low (0 = AT low)
            price_accel     — 1-month momentum minus 3-month momentum (acceleration)
        """
        if "Close" not in df.columns:
            return df
        c = df["Close"]

        # Skip last 5 days (standard Jegadeesh-Titman construction avoids reversal)
        df["reversal_1w"]   = c / c.shift(5).replace(0, np.nan) - 1
        df["momentum_1m"]   = c.shift(5) / c.shift(26).replace(0, np.nan) - 1   # 5→26
        df["momentum_3m"]   = c.shift(5) / c.shift(68).replace(0, np.nan) - 1   # 5→68
        df["momentum_6m"]   = c.shift(5) / c.shift(131).replace(0, np.nan) - 1  # 5→131
        df["momentum_12m"]  = c.shift(5) / c.shift(257).replace(0, np.nan) - 1  # 5→257

        roll252_max = c.rolling(252, min_periods=63).max()
        roll252_min = c.rolling(252, min_periods=63).min()
        df["high_52w_pct"]  = (c / roll252_max.replace(0, np.nan) - 1).clip(-1, 0)
        df["low_52w_pct"]   = (c / roll252_min.replace(0, np.nan) - 1).clip(0, None)

        df["price_accel"]   = df["momentum_1m"] - df["momentum_3m"]

        return df

    def _merge_macro_features(
        self, df: pd.DataFrame, start: str, end: str
    ) -> pd.DataFrame:
        """
        Pull cross-asset / macro time series from yfinance and merge onto spine.

        Tickers fetched:
            SPY   — S&P 500 ETF  (beta, correlation)
            QQQ   — Nasdaq-100   (correlation)
            ^VIX  — CBOE VIX     (fear gauge level)
            ^TNX  — US 10Y yield (rate sensitivity)
            ^IRX  — US 3M yield  (yield-curve slope proxy)
            SOXX  — Semi ETF     (sector relative strength)

        Adds:
            vix_level           — daily VIX close
            yield_10y           — 10-year treasury yield (%)
            yield_curve_slope   — 10Y minus 3M yield (bp proxy)
            spy_beta_60d        — rolling 60-day OLS beta to SPY
            qqq_corr_20d        — rolling 20-day Pearson corr to QQQ returns
            sox_rel_strength    — symbol daily return minus SOXX daily return
        """
        import yfinance as yf

        tickers = {"SPY": None, "QQQ": None, "^VIX": None,
                   "^TNX": None, "^IRX": None, "SOXX": None}

        # Extend lookback by 90d to warm rolling windows
        _start = (pd.Timestamp(start) - pd.Timedelta(days=90)).date().isoformat()

        for tkr in list(tickers):
            try:
                raw = yf.download(tkr, start=_start, end=end,
                                  progress=False, auto_adjust=True)
                if raw is not None and not raw.empty:
                    raw.index = pd.to_datetime(raw.index).tz_localize(None)
                    # Flatten MultiIndex columns if present (yfinance ≥ 0.2.38)
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    tickers[tkr] = raw["Close"].rename(tkr)
            except Exception as exc:
                logger.warning("Macro fetch failed for %s: %s", tkr, exc)

        # ── Align all series to the spine index ──────────────────────────
        def _align(series: pd.Series) -> pd.Series:
            """Forward-fill to trading spine, limit 5 days."""
            combined = series.reindex(series.index.union(df.index)).ffill(limit=5)
            return combined.reindex(df.index)

        sym_ret  = df["log_return"] if "log_return" in df.columns else None
        spy_s    = tickers.get("SPY")
        qqq_s    = tickers.get("QQQ")
        vix_s    = tickers.get("^VIX")
        tnx_s    = tickers.get("^TNX")
        irx_s    = tickers.get("^IRX")
        soxx_s   = tickers.get("SOXX")

        # VIX level
        if vix_s is not None:
            df["vix_level"] = _align(vix_s).values

        # 10Y yield and yield curve slope
        if tnx_s is not None:
            df["yield_10y"] = _align(tnx_s).values
            if irx_s is not None:
                slope = _align(tnx_s) - _align(irx_s)
                df["yield_curve_slope"] = slope.values

        # SPY beta (rolling 60-day OLS)
        if spy_s is not None and sym_ret is not None:
            spy_ret = np.log(spy_s / spy_s.shift(1)).replace(
                [np.inf, -np.inf], np.nan
            )
            spy_ret_aligned = _align(spy_ret)
            cov  = sym_ret.rolling(60, min_periods=20).cov(spy_ret_aligned)
            vari = spy_ret_aligned.rolling(60, min_periods=20).var().replace(0, np.nan)
            df["spy_beta_60d"] = (cov / vari).replace([np.inf, -np.inf], np.nan)

        # QQQ correlation (rolling 20-day)
        if qqq_s is not None and sym_ret is not None:
            qqq_ret = np.log(qqq_s / qqq_s.shift(1)).replace(
                [np.inf, -np.inf], np.nan
            )
            qqq_ret_aligned = _align(qqq_ret)
            df["qqq_corr_20d"] = sym_ret.rolling(20, min_periods=10).corr(
                qqq_ret_aligned
            )

        # Semiconductor sector relative strength
        if soxx_s is not None and sym_ret is not None:
            soxx_ret = np.log(soxx_s / soxx_s.shift(1)).replace(
                [np.inf, -np.inf], np.nan
            )
            soxx_ret_aligned = _align(soxx_ret)
            df["sox_rel_strength"] = (sym_ret - soxx_ret_aligned).replace(
                [np.inf, -np.inf], np.nan
            )

        # Trim warmup rows that crept in from macro fetch expansion
        df = df[df.index >= pd.Timestamp(start)]

        return df

    def _merge_earnings(self, df: pd.DataFrame, sym: str) -> pd.DataFrame:
        """
        Add three earnings-proximity features to the spine:
          earn_days_to_next    — positive int, 9999 if no future date known
          earn_days_since_last — positive int, 9999 if no past date known
          earn_is_week         — int8 flag: 1 if within ±5 calendar days.
        """
        try:
            cal = self.ex.earnings_calendar(sym, limit=20)
            if cal is None or cal.empty:
                return df

            # Normalise earnings date column from Finnhub ('date') or yfinance (index)
            if "date" in cal.columns:
                earn_dates = pd.to_datetime(cal["date"]).dt.tz_localize(None)
            elif "Date" in cal.columns:
                earn_dates = pd.to_datetime(cal["Date"]).dt.tz_localize(None)
            else:
                earn_dates = pd.to_datetime(cal.index).tz_localize(None)

            earn_dates = np.sort(earn_dates.dropna().unique())
            if len(earn_dates) == 0:
                return df

            def _to_next(d: pd.Timestamp) -> int:
                future = earn_dates[earn_dates >= d]
                return int((future[0] - d).days) if len(future) else 9999

            def _since_last(d: pd.Timestamp) -> int:
                past = earn_dates[earn_dates <= d]
                return int((d - past[-1]).days) if len(past) else 9999

            df["earn_days_to_next"]    = [_to_next(d)    for d in df.index]
            df["earn_days_since_last"] = [_since_last(d) for d in df.index]
            df["earn_is_week"]         = (
                (df["earn_days_to_next"] <= 5) | (df["earn_days_since_last"] <= 5)
            ).astype("int8")

        except Exception as exc:
            logger.warning("Earnings merge failed for %s: %s", sym, exc)
        return df
