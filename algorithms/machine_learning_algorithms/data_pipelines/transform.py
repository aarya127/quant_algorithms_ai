"""
transform.py — Unified feature-matrix builder for ML pipelines.

Merges every data source from DataExtractor into a single daily-indexed
DataFrame where each row is one trading day and each column is a named
feature ready for model consumption.

Merge strategy
--------------
  Spine          : technicals()        — OHLCV + computed indicators (daily)
  Derived        : computed in-place   — %B, price/SMA ratios, volume z-score
  Broadcast      : fundamentals()      — financial ratios repeated across dates
  Aggregated     : news()              — daily article count
  Resampled      : sentiment()         — monthly insider MSPR forward-filled
  Snapshot       : options()           — ATM-IV, P/C-OI ratio, max-pain
  Event flags    : earnings_calendar() — proximity & binary earnings flags

Point-in-time note
------------------
Fundamentals and options snapshot columns reflect today's values and are
broadcast statically.  This is safe for live inference / ranking.
For a rigorous backtest, replace these with a point-in-time fundamental
database (e.g. Compustat / WRDS) before training.

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
    # Fundamentals (broadcast, point-in-time caveat)
    "fund_pe_ratio":         "Trailing P/E ratio",
    "fund_forward_pe":       "Forward P/E ratio",
    "fund_pb_ratio":         "Price-to-book ratio",
    "fund_ps_ratio":         "Price-to-sales (TTM)",
    "fund_peg_ratio":        "PEG ratio (PE / EPS growth)",
    "fund_ev_to_ebitda":     "Enterprise value / EBITDA",
    "fund_ev_to_revenue":    "Enterprise value / Revenue",
    "fund_gross_margin":     "Gross margin (0–1)",
    "fund_operating_margin": "Operating margin",
    "fund_net_margin":       "Net profit margin",
    "fund_ebitda_margin":    "EBITDA margin",
    "fund_roe":              "Return on equity",
    "fund_roa":              "Return on assets",
    "fund_revenue_ttm":      "Trailing 12-month revenue",
    "fund_ebitda":           "EBITDA",
    "fund_free_cash_flow":   "Free cash flow",
    "fund_total_cash":       "Cash and equivalents",
    "fund_total_debt":       "Total debt",
    "fund_debt_to_equity":   "Debt-to-equity ratio",
    "fund_current_ratio":    "Current ratio (liquidity)",
    "fund_quick_ratio":      "Quick ratio",
    "fund_revenue_growth":   "YoY revenue growth (TTM)",
    "fund_earnings_growth":  "YoY earnings growth",
    "fund_dividend_yield":   "Dividend yield",
    "fund_beta":             "Beta vs market index",
    "fund_roic":             "Return on invested capital (Finnhub)",
    "fund_roce":             "Return on capital employed (Finnhub)",
    "fund_52w_high":         "52-week high price",
    "fund_52w_low":          "52-week low price",
    "fund_as_of":            "Date the fundamentals snapshot was taken",
    # News
    "news_count":     "Number of news articles indexed on this trading day",
    # Insider sentiment (monthly, forward-filled to daily)
    "insdr_change":   "Net insider buy/sell volume for the month (Finnhub)",
    "insdr_mspr":     "Monthly Share Purchase Ratio — MSPR (Finnhub); negative = net selling",
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
        spine = self.ex.technicals(sym, period=period, interval=interval)
        if spine is None or spine.empty:
            logger.error("No OHLCV data for %s — aborting", sym)
            return pd.DataFrame()

        # Normalise index to tz-naive for consistent joining across sources
        spine.index = pd.to_datetime(spine.index).tz_localize(None)
        spine.index.name = "Date"
        spine["symbol"] = sym

        # Determine date bounds from spine (used for ranged sub-queries)
        start_str = spine.index.min().date().isoformat()
        end_str   = spine.index.max().date().isoformat()

        # ── 2. Derived price / volatility ratios ────────────────────────
        spine = self._add_derived_features(spine)

        # ── 3. Fundamentals (broadcast across all dates) ─────────────────
        spine = self._merge_fundamentals(spine, sym)

        # ── 4. News — daily article count ────────────────────────────────
        spine = self._merge_news(spine, sym, start_str, end_str)

        # ── 5. Insider sentiment (monthly → daily by forward-fill) ───────
        spine = self._merge_sentiment(spine, sym)

        # ── 6. Options snapshot — surface metrics ─────────────────────────
        spine = self._merge_options_snapshot(spine, sym)

        # ── 7. Earnings calendar proximity flags ─────────────────────────
        spine = self._merge_earnings(spine, sym)

        # ── 8. Column ordering: symbol first, then chronological groups ──
        group_order = (
            ["symbol"]
            + [c for c in spine.columns if c in ("Open","High","Low","Close","Volume")]
            + [c for c in spine.columns if c in ("SMA_20","SMA_50","SMA_200","EMA_20",
               "RSI_14","MACD","MACD_signal","MACD_hist","BB_upper","BB_mid","BB_lower",
               "ATR_14","OBV","daily_return","log_return")]
            + [c for c in spine.columns if c.startswith("price_to_") or c in
               ("bb_pct","volume_zscore","high_low_range","overnight_gap")]
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
        """Broadcast all fundamental ratios as constant columns across the spine."""
        try:
            fund = self.ex.fundamentals(sym)
            if fund is None or fund.empty:
                return df

            # Drop metadata / non-numeric columns that don't belong in the matrix
            _drop = {"symbol", "fiscal_period", "fiscal_revenue", "fiscal_net_income",
                     "fiscal_eps", "accruals_ratio", "altman_z", "sloan_ratio",
                     "finnhub_industry", "finnhub_exchange"}
            fund = fund.drop(columns=[c for c in _drop if c in fund.columns], errors="ignore")

            for col in fund.columns:
                df[f"fund_{col}"] = fund[col].iloc[0]

            df["fund_as_of"] = datetime.date.today().isoformat()

        except Exception as exc:
            logger.warning("Fundamentals merge failed for %s: %s", sym, exc)
        return df

    def _merge_news(
        self, df: pd.DataFrame, sym: str, start: str, end: str
    ) -> pd.DataFrame:
        """Left-join daily article counts to the spine (0 on missing days)."""
        try:
            news = self.ex.news(sym, start=start, end=end, limit=500)
            if news is None or news.empty:
                df["news_count"] = 0
                return df

            dates = (
                pd.to_datetime(news["datetime"])
                .dt.tz_localize(None)
                .dt.normalize()
            )
            daily = dates.value_counts().rename_axis("Date").rename("news_count").sort_index()
            df = df.join(daily, how="left")
            df["news_count"] = df["news_count"].fillna(0).astype(int)

        except Exception as exc:
            logger.warning("News merge failed for %s: %s", sym, exc)
            df["news_count"] = 0
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
                monthly.set_index("Date")[["change", "mspr"]]
                .rename(columns={"change": "insdr_change", "mspr": "insdr_mspr"})
            )

            # Join monthly points onto daily spine, then forward-fill within months
            combined = pd.DataFrame(index=df.index).join(monthly, how="left").ffill()
            df["insdr_change"] = combined["insdr_change"]
            df["insdr_mspr"]   = combined["insdr_mspr"]

        except Exception as exc:
            logger.warning("Sentiment merge failed for %s: %s", sym, exc)
        return df

    def _merge_options_snapshot(self, df: pd.DataFrame, sym: str) -> pd.DataFrame:
        """
        Derive four options-surface scalars from the front-expiry chain and
        broadcast them as constant columns across the spine.

        Metrics
        -------
        opt_atm_iv      — IV of the call with strike nearest to last close
        opt_put_call_oi — put OI / call OI (bearish positioning > 1)
        opt_total_oi    — total OI across all strikes in front expiry
        opt_max_pain    — strike that minimises total OI-weighted intrinsic payout
                          to option buyers (widely watched by MM desks)
        """
        try:
            opts = self.ex.options(sym, contract_type="both", enrich_polygon=False)
            if opts is None or opts.empty:
                return df

            last_close = float(df["Close"].iloc[-1])
            front_exp  = opts["expiration"].min() if "expiration" in opts.columns else None

            atm_iv = put_call_oi = total_oi = max_pain = float("nan")

            if front_exp:
                front  = opts[opts["expiration"] == front_exp]
                calls  = front[front["optionType"] == "call"].copy()
                puts   = front[front["optionType"] == "put"].copy()

                # ATM IV — nearest call to current price
                if not calls.empty and "strike" in calls.columns and "impliedVolatility" in calls.columns:
                    nearest_idx = (calls["strike"] - last_close).abs().idxmin()
                    atm_iv = float(calls.loc[nearest_idx, "impliedVolatility"])

                # Put / call OI ratio
                call_oi = float(calls["openInterest"].fillna(0).sum()) if "openInterest" in calls.columns else 0
                put_oi  = float(puts["openInterest"].fillna(0).sum())  if "openInterest" in puts.columns  else 0
                total_oi    = int(call_oi + put_oi)
                put_call_oi = round(put_oi / call_oi, 4) if call_oi > 0 else float("nan")

                # Max-pain: iterate strikes, find minimum aggregate intrinsic loss to buyers
                if "openInterest" in front.columns:
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
