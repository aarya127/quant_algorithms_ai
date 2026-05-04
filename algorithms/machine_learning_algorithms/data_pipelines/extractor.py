"""
extractor.py — Single-file market data extraction layer.

Primary source : yfinance (no key, no rate limit, comprehensive)
Supplementary  : Finnhub    (60 req/min  — news, sentiment, profile enrichment)
                 Fiscal.ai  (250 req/day — structured fundamentals & filings)
                 Polygon    (5 req/min   — options chain with IV & greeks)
                 Alpaca     (200 req/min — SIP-tape OHLCV, news, calendar via alpaca-py SDK)
Excluded       : AlphaVantage (25 req/day shared — reserved for macro calls elsewhere)

All public methods return pandas DataFrames.

Quick start
-----------
    from algorithms.machine_learning_algorithms.data_pipelines.extractor import DataExtractor

    ex = DataExtractor()
    bars  = ex.ohlcv("AAPL", start="2024-01-01", end="2024-12-31")
    news  = ex.news("AAPL", start="2024-01-01", end="2024-01-31")
    funds = ex.fundamentals("AAPL")
    opts  = ex.options("AAPL")

    # Run live connectivity test
    ex.test_apis("AAPL")

Key resolution (env var takes priority over keys.txt):
    FINNHUB_API_KEY
    POLYGON_API_KEY
    FISCAL_AI_API_KEY
    ALPACA_API_KEY    — key ID (embedded in the comment line in keys.txt)
    ALPACA_SECRET_KEY — must be set via env var (not stored in keys.txt)
"""

import datetime
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
import yfinance as yf

# alpaca-py SDK — install with: pip install alpaca-py
try:
    from alpaca.data.historical import StockHistoricalDataClient, NewsClient
    from alpaca.data.requests import StockBarsRequest, NewsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest
    _ALPACA_SDK_AVAILABLE = True
except ImportError:
    _ALPACA_SDK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("DataExtractor")

# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]   # quant_algorithms_ai/


def _read_keys_txt() -> dict:
    """
    Parse keys.txt into a dict of {lower_label: value}.

    Two formats are supported:
      Format A — key on the line AFTER the comment (most providers):
          # finnhub - 60 api calls/minute

      Format B — key EMBEDDED in the comment line itself (Alpaca):
          # alpaca code - <your-key-id> - 200 calls/min
    """
    keys: dict = {}
    path = _REPO_ROOT / "keys.txt"
    if not path.exists():
        return keys
    lines = path.read_text().splitlines()
    # Regex for a UUID-like or long hex key embedded after the provider name in a comment
    _embedded = re.compile(r'[\w.-]*alpaca[\w\s]*[-–]\s*([0-9a-f-]{20,})', re.I)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        label = stripped.lstrip("#").strip().lower()
        # Format B: key embedded in comment (e.g. Alpaca key ID in the comment line itself)
        m = _embedded.search(line)   # .search not .match — pattern may not start at pos 0
        if m:
            keys[label] = m.group(1).strip()
            continue
        # Format A: key on next line
        if i + 1 < len(lines):
            value = lines[i + 1].strip()
            if value and not value.startswith("#"):
                keys[label] = value
    return keys


_KEYS = _read_keys_txt()


def _get_key(env_var: str, keys_label: str) -> str:
    """Return key from env var first, then keys.txt, else empty string."""
    return os.environ.get(env_var) or _KEYS.get(keys_label, "")


# ---------------------------------------------------------------------------
# Per-provider key resolution
# ---------------------------------------------------------------------------
def _finnhub_key() -> str:
    return _get_key("FINNHUB_API_KEY", "finhub - 60 api calls/minute")

def _polygon_key() -> str:
    # Polygon and Massive share the same secret key (0xdCjEkgJ_g8RMrlnGgoO8bm5PYUpUO6)
    # Rate limit: 5 calls/minute (12s between requests)
    return _get_key("POLYGON_API_KEY", "massive - 5 api calls/minute")

def _massive_s3_creds() -> tuple[str, str]:
    """Return (access_key_id, secret_access_key) for Massive S3 flatfiles bucket.

    Activate dataset subscriptions at massive.com portal before calling get_object.
    Rate limit shared with Polygon: 5 calls/minute.
    """
    key_id = _get_key("MASSIVE_ACCESS_KEY_ID", "massive - 5 api calls/minute")
    secret  = _get_key("MASSIVE_SECRET_KEY",    "massive - 5 api calls/minute")
    return key_id, secret

def _fiscal_ai_key() -> str:
    return _get_key("FISCAL_AI_API_KEY", "fiscal.ai - 250 calls/day")

def _alphavantage_key() -> str:
    return _get_key("ALPHAVANTAGE_API_KEY", "alphavantage - 25 requests per day")

def _alpaca_key() -> str:
    val = os.environ.get("ALPACA_API_KEY")
    if val:
        return val
    # Keys.txt stores the key ID embedded in the comment line; _read_keys_txt() handles both formats
    for label, v in _KEYS.items():
        if "alpaca" in label and "massive" not in label:
            return v
    return ""

def _alpaca_secret() -> str:
    return os.environ.get("ALPACA_SECRET_KEY", "")


# ---------------------------------------------------------------------------
# Fiscal.ai — free-tier supported companies (EXCHANGE_SYMBOL format)
# Updated: v1.1.4 free plan, 250 calls/day
# ---------------------------------------------------------------------------
_FISCAL_AI_COMPANIES: dict = {
    "MSFT": "NASDAQ_MSFT", "NVDA": "NASDAQ_NVDA", "AMZN": "NASDAQ_AMZN",
    "GOOG": "NASDAQ_GOOG", "GOOGL": "NASDAQ_GOOG", "TSLA": "NASDAQ_TSLA",
    "LLY":  "NYSE_LLY",    "AVGO": "NASDAQ_AVGO", "V":    "NYSE_V",
    "MA":   "NYSE_MA",     "PG":   "NYSE_PG",     "NFLX": "NASDAQ_NFLX",
    "MCD":  "NYSE_MCD",    "AMGN": "NASDAQ_AMGN", "CAT":  "NYSE_CAT",
    "UBER": "NYSE_UBER",   "MDT":  "NYSE_MDT",    "DUK":  "NYSE_DUK",
    "EQIX": "NASDAQ_EQIX", "BRO":  "NYSE_BRO",    "ZM":   "NASDAQ_ZM",
    "MKC":  "NYSE_MKC",    "RYAN": "NYSE_RYAN",   "MOH":  "NYSE_MOH",
    "CFG":  "NYSE_CFG",    "JPM":  "NYSE_JPM",
}


def _fiscal_ai_company_id(symbol: str) -> Optional[str]:
    """Return the Fiscal.ai company identifier (EXCHANGE_SYMBOL) for a ticker.
    Returns None if the symbol is not on the free-tier supported list.
    """
    return _FISCAL_AI_COMPANIES.get(symbol.upper())


# ---------------------------------------------------------------------------
# DataExtractor
# ---------------------------------------------------------------------------
class DataExtractor:
    """
    Unified data extraction interface.

    yfinance is the backbone for every data type.  Where secondary sources
    add richer data (options greeks, structured filings, company news) they
    are merged on top of the yfinance result.  If a secondary source fails
    (no key, rate limit, network error) the yfinance baseline is returned
    unchanged — extraction never hard-fails.
    """

    # ------------------------------------------------------------------
    # Lazy SDK client cache (one instance per process)
    # ------------------------------------------------------------------
    _stock_client: Optional["StockHistoricalDataClient"] = None
    _news_client:  Optional["NewsClient"] = None
    _trade_client: Optional["TradingClient"] = None

    def _alpaca_clients(self):
        """
        Return (stock_client, news_client, trade_client) or (None, None, None)
        if the SDK is not installed or credentials are missing.
        """
        if not _ALPACA_SDK_AVAILABLE:
            return None, None, None
        key    = _alpaca_key()
        secret = _alpaca_secret()
        if not key or not secret:
            return None, None, None
        if self._stock_client is None:
            self._stock_client = StockHistoricalDataClient(api_key=key, secret_key=secret)
        if self._news_client is None:
            self._news_client = NewsClient(api_key=key, secret_key=secret)
        if self._trade_client is None:
            self._trade_client = TradingClient(api_key=key, secret_key=secret)
        return self._stock_client, self._news_client, self._trade_client

    # -----------------------------------------------------------------------
    # OHLCV
    # -----------------------------------------------------------------------

    def ohlcv(
        self,
        symbol: str,
        start: str = "",
        end:   str = "",
        period:   str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        OHLCV price bars.  Returns DataFrame with columns:
        Open, High, Low, Close, Volume, Dividends, Stock Splits

        Parameters
        ----------
        start / end  : 'YYYY-MM-DD' — if both supplied, used instead of period.
        period       : yfinance period string ('1d','5d','1mo','3mo','6mo',
                       '1y','2y','5y','10y','ytd','max') — used when start/end absent.
        interval     : bar size ('1m','5m','15m','30m','60m','1h','1d','1wk','1mo').

        Primary  : yfinance (adjusted, full history)
        Fallback : Alpaca REST bars (SIP tape) — tried when ALPACA_SECRET_KEY set.
        """
        df = self._yf_ohlcv(symbol, start, end, period, interval)
        if df is not None and not df.empty:
            return df

        logger.warning("yfinance OHLCV failed for %s, trying Polygon SIP tape", symbol)
        if start and end:
            try:
                df = self._polygon_ohlcv_raw(symbol, start, end)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass

        logger.warning("Polygon OHLCV failed for %s, trying Alpaca", symbol)
        df = self._alpaca_ohlcv(symbol, start, end, interval)
        if df is not None and not df.empty:
            return df

        return pd.DataFrame()

    def _yf_ohlcv(self, symbol, start, end, period, interval) -> Optional[pd.DataFrame]:
        try:
            t = yf.Ticker(symbol)
            if start and end:
                df = t.history(start=start, end=end, interval=interval)
            else:
                df = t.history(period=period, interval=interval)
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            return df
        except Exception as exc:
            logger.warning("yfinance OHLCV error for %s: %s", symbol, exc)
            return None

    def _alpaca_ohlcv(self, symbol, start, end, interval) -> Optional[pd.DataFrame]:
        stock_client, _, _ = self._alpaca_clients()
        if stock_client is None:
            return None
        _tf_map = {
            "1d":    TimeFrame.Day,
            "1h":    TimeFrame.Hour,
            "1min":  TimeFrame.Minute,
            "5min":  TimeFrame(5,  TimeFrameUnit.Minute),
            "15min": TimeFrame(15, TimeFrameUnit.Minute),
            "30min": TimeFrame(30, TimeFrameUnit.Minute),
        }
        tf = _tf_map.get(interval, TimeFrame.Day)
        try:
            from datetime import datetime as _dt
            req = StockBarsRequest(
                symbol_or_symbols=[symbol.upper()],
                timeframe=tf,
                start=_dt.fromisoformat(start),
                end=_dt.fromisoformat(end),
                adjustment="all",
                feed="sip",
            )
            bars = stock_client.get_stock_bars(req)
            df = bars.df
            if df.empty:
                return None
            # SDK returns MultiIndex (symbol, timestamp) — drop symbol level
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel(0)
            df.index.name = "Date"
            df.columns = [c.capitalize() for c in df.columns]
            return df
        except Exception as exc:
            logger.warning("Alpaca SDK OHLCV error for %s: %s", symbol, exc)
            return None

    # -----------------------------------------------------------------------
    # PROFILE
    # -----------------------------------------------------------------------

    def profile(self, symbol: str) -> pd.DataFrame:
        """
        Company profile metadata.  Returns 1-row DataFrame with columns:
        name, sector, industry, country, exchange, market_cap, employees,
        website, description, source_primary, finnhub_exchange (if enriched)

        Primary  : yfinance (.info dict)
        Enriched : Finnhub  (ipo_date, country, exchange details)
        """
        row = self._yf_profile(symbol)
        fh  = self._finnhub_profile(symbol)
        if fh:
            # Fill any None fields with Finnhub values
            for k, v in fh.items():
                if row.get(k) is None and v:
                    row[k] = v
        row["symbol"] = symbol.upper()
        return pd.DataFrame([row])

    def _yf_profile(self, symbol) -> dict:
        try:
            info = yf.Ticker(symbol).info
            return {
                "name":        info.get("longName") or info.get("shortName"),
                "sector":      info.get("sector"),
                "industry":    info.get("industry"),
                "country":     info.get("country"),
                "exchange":    info.get("exchange"),
                "market_cap":  info.get("marketCap"),
                "employees":   info.get("fullTimeEmployees"),
                "website":     info.get("website"),
                "description": info.get("longBusinessSummary"),
            }
        except Exception as exc:
            logger.warning("yfinance profile error for %s: %s", symbol, exc)
            return {}

    def _finnhub_profile(self, symbol) -> Optional[dict]:
        key = _finnhub_key()
        if not key:
            return None
        url = "https://finnhub.io/api/v1/stock/profile2"
        try:
            resp = requests.get(url, params={"symbol": symbol, "token": key}, timeout=10)
            resp.raise_for_status()
            d = resp.json()
            return {
                "name":             d.get("name"),
                "country":          d.get("country"),
                "finnhub_exchange":  d.get("exchange"),
                "ipo_date":         d.get("ipo"),
                "market_cap":       d.get("marketCapitalization"),
                "employees":        d.get("employeeTotal"),
                "finnhub_industry": d.get("finnhubIndustry"),
                "website":          d.get("weburl"),
            }
        except Exception as exc:
            logger.warning("Finnhub profile error for %s: %s", symbol, exc)
            return None

    # -----------------------------------------------------------------------
    # FUNDAMENTALS
    # -----------------------------------------------------------------------

    def fundamentals(self, symbol: str) -> pd.DataFrame:
        """
        Financial ratios and key metrics.  Returns 1-row DataFrame.

        Primary  : yfinance (.info dict — comprehensive, free)
        Enriched : Finnhub basic_financials (ROIC, ROCE, multi-period metrics)
        Enriched : Fiscal.ai (structured accounting signals — when key available)
        """
        row: dict = {"symbol": symbol.upper()}
        row.update(self._yf_fundamentals(symbol))
        row.update(self._finnhub_fundamentals(symbol) or {})
        fiscal = self._fiscal_ai_fundamentals(symbol)
        if fiscal:
            for k, v in fiscal.items():
                if row.get(k) is None:
                    row[k] = v
        return pd.DataFrame([row])

    def _yf_fundamentals(self, symbol) -> dict:
        try:
            info = yf.Ticker(symbol).info
            return {
                # Valuation
                "pe_ratio":          info.get("trailingPE"),
                "forward_pe":        info.get("forwardPE"),
                "pb_ratio":          info.get("priceToBook"),
                "ps_ratio":          info.get("priceToSalesTrailing12Months"),
                "peg_ratio":         info.get("pegRatio"),
                "ev_to_ebitda":      info.get("enterpriseToEbitda"),
                "ev_to_revenue":     info.get("enterpriseToRevenue"),
                # Margins
                "gross_margin":      info.get("grossMargins"),
                "operating_margin":  info.get("operatingMargins"),
                "net_margin":        info.get("profitMargins"),
                "ebitda_margin":     info.get("ebitdaMargins"),
                # Returns
                "roe":               info.get("returnOnEquity"),
                "roa":               info.get("returnOnAssets"),
                # Financials
                "revenue_ttm":       info.get("totalRevenue"),
                "ebitda":            info.get("ebitda"),
                "free_cash_flow":    info.get("freeCashflow"),
                "total_cash":        info.get("totalCash"),
                "total_debt":        info.get("totalDebt"),
                "debt_to_equity":    info.get("debtToEquity"),
                "current_ratio":     info.get("currentRatio"),
                "quick_ratio":       info.get("quickRatio"),
                # Growth
                "revenue_growth":    info.get("revenueGrowth"),
                "earnings_growth":   info.get("earningsGrowth"),
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
                # Dividends
                "dividend_yield":    info.get("dividendYield"),
                "payout_ratio":      info.get("payoutRatio"),
                # Other
                "beta":              info.get("beta"),
                "shares_outstanding":info.get("sharesOutstanding"),
                "short_ratio":       info.get("shortRatio"),
            }
        except Exception as exc:
            logger.warning("yfinance fundamentals error for %s: %s", symbol, exc)
            return {}

    def _finnhub_fundamentals(self, symbol) -> Optional[dict]:
        key = _finnhub_key()
        if not key:
            return None
        url = "https://finnhub.io/api/v1/stock/metric"
        try:
            resp = requests.get(url, params={"symbol": symbol, "metric": "all", "token": key}, timeout=10)
            resp.raise_for_status()
            m = resp.json().get("metric") or {}
            return {
                "roic":                    m.get("roicTTM"),
                "roce":                    m.get("roiTTM"),       # Finnhub exposes ROI ≈ ROCE
                "ev_to_ebitda_finnhub":    m.get("evToEbitdaTTM"),
                "revenue_3y_growth_pa":    m.get("revenueGrowth3Y"),
                "eps_growth_3y_pa":        m.get("epsGrowth3Y"),
                "gross_margin_5y_avg":     m.get("grossMargin5Y"),
                "net_margin_5y_avg":       m.get("netMargin5Y"),
                "current_ratio_annual":    m.get("currentRatioAnnual"),
                "debt_to_equity_annual":   m.get("totalDebt/totalEquityAnnual"),
                "revenue_per_share_ttm":   m.get("revenuePerShareTTM"),
                "bookvalue_per_share":     m.get("bookValuePerShareAnnual"),
                "52w_high":                m.get("52WeekHigh"),
                "52w_low":                 m.get("52WeekLow"),
            }
        except Exception as exc:
            logger.warning("Finnhub fundamentals error for %s: %s", symbol, exc)
            return None

    def _fiscal_ai_fundamentals(self, symbol) -> Optional[dict]:
        key = _fiscal_ai_key()
        if not key:
            return None
        company_id = _fiscal_ai_company_id(symbol)
        if not company_id:
            logger.debug("Fiscal.ai: %s not in free-tier company list — skipping", symbol)
            return None
        url = f"https://api.fiscal.ai/v1/company/{company_id}/financials"
        headers = {"Authorization": f"Bearer {key}"}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return {
                "fiscal_period":    data.get("period"),
                "fiscal_revenue":   data.get("revenue"),
                "fiscal_net_income":data.get("net_income"),
                "fiscal_eps":       data.get("eps"),
                "accruals_ratio":   data.get("accruals_ratio"),
                "altman_z":         data.get("altman_z"),
                "sloan_ratio":      data.get("sloan_ratio"),
            }
        except Exception as exc:
            logger.warning("Fiscal.ai fundamentals error for %s: %s", symbol, exc)
            return None

    # -----------------------------------------------------------------------
    # HISTORICAL FUNDAMENTALS  (point-in-time quarterly time series)
    # -----------------------------------------------------------------------

    def historical_fundamentals(self, symbol: str) -> pd.DataFrame:
        """
        Historical fundamental metrics derived from yfinance financial statements.

        Strategy (two tiers, merged so quarterly overrides annual):
          • Quarterly (4–5 quarters back) — income/CF metrics are TTM-summed;
            balance-sheet metrics use the closest quarterly snapshot.
          • Annual (4 fiscal years back) — fills dates before quarterly data
            begins; each annual value is broadcast across the whole fiscal year.

        Every row carries the exact same column set so merge_asof in transform
        simply matches each trading day to the most recently available filing.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (period-end, tz-naive, ascending, ~20–25 rows total)
            Columns: fund_rev_ttm, fund_gross_profit_ttm, fund_net_income_ttm,
                     fund_eps_ttm, fund_fcf_ttm, fund_gross_margin,
                     fund_operating_margin, fund_net_margin, fund_roe, fund_roa,
                     fund_debt_to_equity, fund_current_ratio,
                     fund_total_assets, fund_total_debt, fund_cash,
                     fund_rev_growth_yoy, fund_quarter_end
        """
        try:
            t = yf.Ticker(symbol)
            q_inc = t.quarterly_income_stmt
            q_bal = t.quarterly_balance_sheet
            q_cf  = t.quarterly_cashflow
            a_inc = t.income_stmt
            a_bal = t.balance_sheet
            a_cf  = t.cashflow

            def _safe(val) -> float:
                try:
                    v = float(val)
                    return v if not pd.isna(v) else float("nan")
                except (TypeError, ValueError):
                    return float("nan")

            def _ratio(num, den) -> float:
                n, d = _safe(num), _safe(den)
                if pd.isna(n) or pd.isna(d) or d == 0:
                    return float("nan")
                return n / d

            def _first_col(frame: pd.DataFrame, *names) -> "pd.Series":
                for name in names:
                    if name in frame.columns:
                        return frame[name]
                return pd.Series(float("nan"), index=frame.index)

            def _prep(raw) -> pd.DataFrame:
                """Transpose raw yfinance statement → rows=dates, cols=metrics."""
                if raw is None or raw.empty:
                    return pd.DataFrame()
                return raw.T.sort_index()

            def _build_rows(inc, bal, cf, ttm_window: int) -> list:
                """
                Build one metrics-dict per period row.

                ttm_window=4  → quarterly (sum 4 quarters)
                ttm_window=1  → annual    (one year = full period)
                """
                rows = []
                for i, date in enumerate(inc.index):
                    w_inc = inc.iloc[max(0, i - (ttm_window - 1)): i + 1]
                    w_cf  = cf.iloc[max(0, i - (ttm_window - 1)): i + 1] if not cf.empty else pd.DataFrame()
                    bal_row = (
                        bal.loc[date]
                        if not bal.empty and date in bal.index
                        else pd.Series(dtype=float)
                    )

                    def ttm(*names) -> float:
                        col  = _first_col(w_inc, *names)
                        vals = col.dropna()
                        return float(vals.sum()) if not vals.empty else float("nan")

                    def cf_ttm(*names) -> float:
                        col  = _first_col(w_cf, *names)
                        vals = col.dropna()
                        return float(vals.sum()) if not vals.empty else float("nan")

                    def bs(*names) -> float:
                        if bal_row.empty:
                            return float("nan")
                        for name in names:
                            if name in bal_row.index:
                                return _safe(bal_row[name])
                        return float("nan")

                    rev     = ttm("Total Revenue",       "TotalRevenue")
                    gross   = ttm("Gross Profit",        "GrossProfit")
                    op_inc  = ttm("Operating Income",    "OperatingIncome", "EBIT")
                    net_inc = ttm("Net Income",          "NetIncome")
                    eps     = ttm("Diluted EPS",         "Basic EPS",
                                  "EPS Diluted",         "DilutedEPS")
                    op_cf   = cf_ttm("Operating Cash Flow",  "OperatingCashFlow",
                                     "Total Cash From Operating Activities")
                    capex   = cf_ttm("Capital Expenditure",  "CapitalExpenditure",
                                     "Capital Expenditures")
                    equity  = bs("Stockholders Equity",  "StockholdersEquity",
                                 "Total Equity Gross Minority Interest")
                    assets  = bs("Total Assets",         "TotalAssets")
                    debt    = bs("Total Debt",           "TotalDebt",
                                 "Long Term Debt",       "LongTermDebt")
                    cash    = bs("Cash And Cash Equivalents", "CashAndCashEquivalents",
                                 "Cash",                 "Cash Equivalents")
                    curr_a  = bs("Current Assets",       "TotalCurrentAssets",
                                 "Total Current Assets")
                    curr_l  = bs("Current Liabilities",  "TotalCurrentLiabilities",
                                 "Total Current Liabilities")

                    fcf = (
                        (op_cf + capex)
                        if not (pd.isna(op_cf) or pd.isna(capex))
                        else float("nan")
                    )

                    rows.append({
                        "fund_rev_ttm":          rev,
                        "fund_gross_profit_ttm": gross,
                        "fund_net_income_ttm":   net_inc,
                        "fund_eps_ttm":          eps,
                        "fund_fcf_ttm":          fcf,
                        "fund_gross_margin":     _ratio(gross,   rev),
                        "fund_operating_margin": _ratio(op_inc,  rev),
                        "fund_net_margin":       _ratio(net_inc, rev),
                        "fund_roe":              _ratio(net_inc, equity),
                        "fund_roa":              _ratio(net_inc, assets),
                        "fund_debt_to_equity":   _ratio(debt,    equity),
                        "fund_current_ratio":    _ratio(curr_a,  curr_l),
                        "fund_total_assets":     assets,
                        "fund_total_debt":       debt,
                        "fund_cash":             cash,
                        "fund_quarter_end":      (
                            str(date.date()) if hasattr(date, "date") else str(date)
                        ),
                    })
                return rows

            def _to_timed_df(rows: list, index) -> pd.DataFrame:
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows, index=index)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "Date"
                return df

            # ── Quarterly tier ────────────────────────────────────────────
            q_rows = pd.DataFrame()
            q_prep = _prep(q_inc)
            if not q_prep.empty:
                rows = _build_rows(q_prep, _prep(q_bal), _prep(q_cf), ttm_window=4)
                q_rows = _to_timed_df(rows, q_prep.index)
                if not q_rows.empty:
                    q_rows["fund_rev_growth_yoy"] = (
                        q_rows["fund_rev_ttm"]
                        .pct_change(4)
                        .replace([float("inf"), float("-inf")], float("nan"))
                    )

            # ── Annual tier (fills earlier dates) ─────────────────────────
            a_rows = pd.DataFrame()
            a_prep = _prep(a_inc)
            if not a_prep.empty:
                rows = _build_rows(a_prep, _prep(a_bal), _prep(a_cf), ttm_window=1)
                a_rows = _to_timed_df(rows, a_prep.index)
                if not a_rows.empty:
                    a_rows["fund_rev_growth_yoy"] = (
                        a_rows["fund_rev_ttm"]
                        .pct_change(1)
                        .replace([float("inf"), float("-inf")], float("nan"))
                    )

            if q_rows.empty and a_rows.empty:
                return pd.DataFrame()

            # Merge: quarterly wins for overlapping dates
            combined = pd.concat([a_rows, q_rows])
            combined = combined[~combined.index.duplicated(keep="last")]
            return combined.sort_index()

        except Exception as exc:
            logger.warning("historical_fundamentals error for %s: %s", symbol, exc)
            return pd.DataFrame()

    # -----------------------------------------------------------------------
    # NEWS
    # -----------------------------------------------------------------------

    def news(
        self,
        symbol: str,
        start:  str = "",
        end:    str = "",
        limit:  int = 50,
    ) -> pd.DataFrame:
        """
        Company news articles.  Returns DataFrame with columns:
        datetime, headline, summary, source, url, sentiment, provider

        Primary  : Finnhub company_news  (most reliable company-specific feed)
        Fallback : yfinance .news          (no date range; less structured)
        Fallback : Alpaca news REST        (requires Alpaca secret key)
        """
        if not start:
            start = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
        if not end:
            end = datetime.date.today().isoformat()

        df = self._finnhub_news(symbol, start, end, limit)
        if df is not None and not df.empty:
            return df

        logger.warning("Finnhub news unavailable for %s, falling back to yfinance", symbol)
        df = self._yf_news(symbol, limit)
        if df is not None and not df.empty:
            return df

        logger.warning("yfinance news failed for %s, trying Alpaca", symbol)
        return self._alpaca_news(symbol, start, end, limit) or pd.DataFrame()

    def _finnhub_news(self, symbol, start, end, limit) -> Optional[pd.DataFrame]:
        key = _finnhub_key()
        if not key:
            return None

        import pickle
        _CACHE_DIR  = Path(__file__).parent / ".pipeline_cache"
        _CACHE_DIR.mkdir(exist_ok=True)
        _cache_path = _CACHE_DIR / f"fh_news_{symbol.upper()}_{start}_{end}.pkl"
        _now        = time.time()
        if _cache_path.exists() and (_now - _cache_path.stat().st_mtime) < 86400:
            try:
                cached = pickle.loads(_cache_path.read_bytes())
                if cached is not None:
                    logger.debug("Finnhub cache hit: %s (%s → %s)", symbol, start, end)
                    return cached
            except Exception:
                pass

        url = "https://finnhub.io/api/v1/company-news"
        try:
            resp = requests.get(
                url,
                # "from" is a Python reserved word so pass params as a dict literal
                params={"symbol": symbol, "from": start, "to": end, "token": key},
                timeout=10,
            )
            resp.raise_for_status()
            articles = resp.json()
            if not articles:
                return None
            # Filter strictly to the requested date range before applying limit
            df = pd.DataFrame(articles)
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
            start_ts = pd.Timestamp(start, tz="UTC")
            end_ts   = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
            df = df[(df["datetime"] >= start_ts) & (df["datetime"] < end_ts)]
            df = df.head(limit)
            if df.empty:
                return None
            df["provider"] = "finnhub"
            cols = ["datetime", "headline", "summary", "source", "url", "provider"]
            result = df[[c for c in cols if c in df.columns]]
            try:
                _cache_path.write_bytes(pickle.dumps(result))
            except Exception:
                pass
            return result
        except Exception as exc:
            logger.warning("Finnhub news error for %s: %s", symbol, exc)
            return None

    def _yf_news(self, symbol, limit) -> Optional[pd.DataFrame]:
        try:
            articles = yf.Ticker(symbol).news[:limit]
            if not articles:
                return None
            rows = []
            for a in articles:
                rows.append({
                    "datetime": pd.to_datetime(a.get("providerPublishTime"), unit="s", utc=True),
                    "headline": a.get("title"),
                    "summary":  None,
                    "source":   a.get("publisher"),
                    "url":      a.get("link"),
                    "provider": "yfinance",
                })
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.warning("yfinance news error for %s: %s", symbol, exc)
            return None

    def _alpaca_news(self, symbol, start, end, limit) -> Optional[pd.DataFrame]:
        _, news_client, _ = self._alpaca_clients()
        if news_client is None:
            return None
        try:
            from datetime import datetime as _dt
            req = NewsRequest(
                symbols=[symbol.upper()],
                start=_dt.fromisoformat(start),
                end=_dt.fromisoformat(end),
                limit=min(limit, 50),   # SDK max per page
            )
            result = news_client.get_news(req)
            articles = result.news
            if not articles:
                return None
            rows = [{
                "datetime": pd.to_datetime(a.created_at),
                "headline": a.headline,
                "summary":  a.summary,
                "source":   a.source,
                "url":      a.url,
                "provider": "alpaca",
            } for a in articles]
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.warning("Alpaca SDK news error for %s: %s", symbol, exc)
            return None

    # -----------------------------------------------------------------------
    # NEWS SENTIMENT HISTORY  (multi-source, daily aggregation)
    # -----------------------------------------------------------------------

    def news_sentiment_history(
        self,
        symbol:      str,
        start:       str = "",
        end:         str = "",
        use_finbert: bool = False,
    ) -> pd.DataFrame:
        """
        Daily news sentiment scores aggregated from multiple sources.

        Returns a DataFrame indexed by date with columns:
            news_sent_score    — blended daily sentiment score (-1 to +1) across all scored sources
            news_sent_av       — Alpha Vantage relevance-weighted daily score (-1 to +1)
            news_sent_polygon  — Polygon / Alpaca AI-insight daily score (-1 to +1)
            news_sent_finnhub  — Finnhub headlines scored by FinBERT P(pos)-P(neg) (-1 to +1)
            news_articles      — total article count for that day (all sources)

        Parameters
        ----------
        use_finbert : bool, default False
            When True, fetch raw article text from all sources and run
            ProsusAI/finbert on each headline+summary.  Score =
            P(positive) - P(negative).  Requires ``transformers`` and
            ``torch`` to be installed.  Adds ~5-15 s model load time and
            ~0.1 s per article; use for offline / research runs.

            When False (default), use API pre-scores from Polygon insights
            and Alpha Vantage — fast, no extra dependencies.

        Source priority (use_finbert=False)
        ------------------------------------
        1. Polygon /v2/reference/news — AI-scored ``insights`` per ticker
           (labeled "massive" in keys.txt — the key starting with 0x...)
        2. Alpha Vantage NEWS_SENTIMENT — ticker-level scores + relevance weights
        3. Finnhub company-news — article count only (no pre-scored sentiment)
        """
        if not start:
            start = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
        if not end:
            end = datetime.date.today().isoformat()

        # ------------------------------------------------------------------
        # Fast path: FinBERT NLP on raw text from all sources
        # ------------------------------------------------------------------
        if use_finbert:
            try:
                from sentiment.finbert import FinBertAnalyzer
                daily_fb = FinBertAnalyzer().analyze(symbol, start, end)
                if daily_fb is not None and not daily_fb.empty:
                    # Rename to match the expected downstream schema
                    daily_fb = daily_fb.rename(columns={
                        "finbert_score": "news_sent_score",
                        "article_count": "news_articles",
                    })
                    logger.info("FinBERT sentiment: %d trading days for %s", len(daily_fb), symbol)
                    return daily_fb[["news_sent_score", "news_articles"]]
            except ImportError:
                logger.warning("use_finbert=True but 'transformers' not installed — falling back to API scores")
            except Exception as exc:
                logger.warning("FinBERT sentiment error for %s: %s — falling back", symbol, exc)

        # ------------------------------------------------------------------
        # Default path: API pre-scores (fast, no ML inference)
        # Per-source frames kept separate to produce per-source columns.
        # ------------------------------------------------------------------
        poly_frames: list[pd.DataFrame] = []   # Polygon + Alpaca AI insights
        av_frames:   list[pd.DataFrame] = []   # Alpha Vantage relevance-weighted

        poly = self._polygon_news_sentiment(symbol, start, end)
        if poly is not None and not poly.empty:
            poly_frames.append(poly)
            logger.info("Polygon news sentiment: %d articles for %s", len(poly), symbol)

        # AV caps results at 1000 per call. For windows > 90 days, batch by
        # 90-day chunks so high-volume tickers (NVDA: ~10 articles/day) stay
        # under the 1000-article cap and get full per-batch coverage.
        # Stop early if rate-limited so we don't waste quota on subsequent batches.
        _av_start_dt = datetime.date.fromisoformat(start)
        _av_end_dt   = datetime.date.fromisoformat(end)
        _av_window   = 30  # days per batch — keeps NVDA (≈11 articles/day) under AV's 1000-article cap
        _av_cursor   = _av_start_dt
        _av_total    = 0
        _av_batches  = 0
        _av_rl_hit   = False
        while _av_cursor < _av_end_dt:
            _b_end   = min(_av_cursor + datetime.timedelta(days=_av_window - 1), _av_end_dt)
            _av_batches += 1
            # Pass allow_stale=True so a rate-limited call still returns cached data
            av_batch = self._av_news_sentiment(symbol, _av_cursor.isoformat(), _b_end.isoformat(),
                                               allow_stale=True)
            if av_batch is not None and not av_batch.empty:
                av_frames.append(av_batch)
                _av_total += len(av_batch)
                logger.info("  AV batch %d (%s → %s): %d articles",
                            _av_batches, _av_cursor, _b_end, len(av_batch))
            elif av_batch is None and not _av_rl_hit:
                # None with no cache available → live rate-limit, no fallback
                _av_rl_hit = True
                logger.warning("AlphaVantage rate-limit reached after %d batches, no cache available — skipping batch",
                               _av_batches)
            _av_cursor = _b_end + datetime.timedelta(days=1)
        if _av_total:
            logger.info("AlphaVantage news sentiment: %d articles for %s (%d batches)",
                        _av_total, symbol, _av_batches)

        alp = self._alpaca_news_sentiment(symbol, start, end)
        if alp is not None and not alp.empty:
            poly_frames.append(alp)  # same AI-insight format as Polygon
            logger.info("Alpaca news sentiment (insights): %d articles for %s", len(alp), symbol)

        # Finnhub: fetch articles then score headlines with FinBERT
        fh_frames:  list[pd.DataFrame] = []   # for article count (all FH articles)
        fhb_frames: list[pd.DataFrame] = []   # FinBERT-scored FH articles
        fh = self._finnhub_news(symbol, start, end, limit=500)
        if fh is not None and not fh.empty:
            fh_prep = fh.copy()
            fh_prep["date"]       = pd.to_datetime(fh_prep["datetime"]).dt.tz_localize(None).dt.normalize()
            fh_prep["sent_score"] = float("nan")
            fh_prep["weight"]     = 1.0

            # Run FinBERT on each Finnhub headline to get a proper sentiment score
            try:
                from sentiment.finbert import FinBertAnalyzer, _load_model
                _load_model()   # load once
                analyzer = FinBertAnalyzer()
                fb_rows: list[dict] = []
                for _, row in fh_prep.iterrows():
                    text = str(row.get("headline", "") or "")
                    summary = str(row.get("summary", "") or "")
                    full_text = (text + " " + summary).strip()
                    _, probs = analyzer.get_sentiment(full_text)
                    if probs:
                        score = probs.get("positive", 0.0) - probs.get("negative", 0.0)
                        fb_rows.append({"date": row["date"], "headline": text,
                                        "sent_score": score, "weight": 1.0})
                if fb_rows:
                    fhb_frames.append(pd.DataFrame(fb_rows))
                    logger.info("Finnhub FinBERT scored: %d articles for %s", len(fb_rows), symbol)
            except ImportError:
                logger.warning("FinBERT not installed — Finnhub articles unscored")
            except Exception as exc:
                logger.warning("Finnhub FinBERT scoring error for %s: %s", symbol, exc)

            keep_cols = ["date", "sent_score", "weight"]
            if "headline" in fh_prep.columns:
                keep_cols.insert(1, "headline")
            fh_frames.append(fh_prep[keep_cols])
            logger.info("Finnhub news count: %d articles for %s", len(fh_prep), symbol)

        all_frames = poly_frames + av_frames + fh_frames
        if not all_frames:
            return pd.DataFrame()

        def _agg_daily(frames: list, score_col: str) -> pd.DataFrame:
            """Weighted-mean daily sentiment + article count from *frames*."""
            if not frames:
                return pd.DataFrame()
            combined = pd.concat(frames, ignore_index=True)
            if "headline" in combined.columns:
                combined = combined.drop_duplicates(subset=["date", "headline"])

            def _agg(g: pd.DataFrame) -> pd.Series:
                valid = g["sent_score"].notna()
                if valid.any():
                    w = g.loc[valid, "weight"].fillna(1.0)
                    s = g.loc[valid, "sent_score"]
                    score = float((s * w).sum() / w.sum())
                else:
                    score = float("nan")
                return pd.Series({score_col: score, "news_articles": len(g)})

            agg = combined.groupby("date").apply(_agg).reset_index()
            agg["date"] = pd.to_datetime(agg["date"])
            return agg.set_index("date").sort_index()

        # Per-source daily aggregations
        poly_daily = _agg_daily(poly_frames, "news_sent_polygon")
        av_daily   = _agg_daily(av_frames,   "news_sent_av")
        fhb_daily  = _agg_daily(fhb_frames,  "news_sent_finnhub")

        # Blended score across all scored sources (API scores + FinBERT-scored Finnhub)
        scored_frames = poly_frames + av_frames + fhb_frames
        blended_daily = _agg_daily(scored_frames if scored_frames else fh_frames, "news_sent_score")

        # Total article count across all sources (including Finnhub)
        total_daily = _agg_daily(all_frames, "_unused")

        if blended_daily.empty and total_daily.empty:
            return pd.DataFrame()

        # Build a unified daily index from the union of all source dates
        all_dates = total_daily.index if not total_daily.empty else blended_daily.index
        daily = pd.DataFrame(index=all_dates)
        daily.index.name = "date"

        if not blended_daily.empty:
            daily = daily.join(blended_daily[["news_sent_score"]], how="left")
        else:
            daily["news_sent_score"] = float("nan")

        if not poly_daily.empty:
            daily = daily.join(poly_daily[["news_sent_polygon"]], how="left")
        else:
            daily["news_sent_polygon"] = float("nan")

        if not av_daily.empty:
            daily = daily.join(av_daily[["news_sent_av"]], how="left")
        else:
            daily["news_sent_av"] = float("nan")

        if not fhb_daily.empty:
            daily = daily.join(fhb_daily[["news_sent_finnhub"]], how="left")
        else:
            daily["news_sent_finnhub"] = float("nan")

        daily["news_articles"] = (
            total_daily["news_articles"].reindex(daily.index).fillna(0).astype(int)
        )
        return daily.sort_index()

    def news_with_scores(
        self,
        symbol:       str,
        start:        str = "",
        end:          str = "",
        max_articles: int = 100,
    ) -> pd.DataFrame:
        """
        Article-level news with pre-computed sentiment scores.

        Returns a DataFrame with columns:
            date        — publication date (normalized, no tz)
            headline    — article title
            source      — publisher name
            provider    — 'alphavantage' | 'finnhub'
            sent_score  — score ∈ [-1, +1]; positive > 0, negative < 0
            label       — 'positive' | 'negative' | 'neutral'
            url         — article URL (if available)

        Sources used (API-limit-safe for UI calls)
        -------------------------------------------
        • Alpha Vantage NEWS_SENTIMENT (1 call, 25/day limit)
        • Finnhub company-news         (1 call, 60/min limit)
        Polygon is intentionally skipped here to preserve the 5 req/min
        budget for the data pipeline.
        """
        if not start:
            start = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
        if not end:
            end = datetime.date.today().isoformat()

        rows: list[dict] = []

        # --- Alpha Vantage (ticker-specific scored sentiment) ---
        av_df = self._av_news_sentiment(symbol, start, end)
        if av_df is not None and not av_df.empty:
            for _, r in av_df.iterrows():
                score = float(r.get("sent_score", 0.0))
                rows.append({
                    "date":       r["date"],
                    "headline":   r.get("headline", ""),
                    "source":     "Alpha Vantage",
                    "provider":   "alphavantage",
                    "sent_score": score,
                    "label":      "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral"),
                    "url":        r.get("url", ""),
                })

        # --- Finnhub (no pre-score, derive from article content if available) ---
        fh_df = self._finnhub_news(symbol, start, end, limit=200)
        if fh_df is not None and not fh_df.empty:
            for _, r in fh_df.iterrows():
                dt = r.get("datetime", pd.NaT)
                date = pd.Timestamp(dt).tz_localize(None).normalize() if dt is not pd.NaT else pd.NaT
                rows.append({
                    "date":       date,
                    "headline":   r.get("headline", ""),
                    "source":     str(r.get("source", "Finnhub")),
                    "provider":   "finnhub",
                    "sent_score": float("nan"),   # no pre-score from Finnhub
                    "label":      "neutral",
                    "url":        r.get("url", ""),
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=False)
        df = df.drop_duplicates(subset=["date", "headline"])
        return df.head(max_articles).reset_index(drop=True)

    # Minimum seconds between Polygon API calls (5 req/min plan)
    _POLYGON_RATE_LIMIT_S: float = 12.0

    def _polygon_news_sentiment(
        self,
        symbol: str,
        start:  str,
        end:    str,
    ) -> Optional[pd.DataFrame]:
        """
        Polygon /v2/reference/news with AI insights.
        Each article may contain per-ticker sentiment + sentiment_reasoning.
        Labeled "massive" in keys.txt (key starts with 0x...).
        Pagination is handled via the ``next_url`` cursor field.
        Rate limit: 5 calls/minute — sleeps 12 s between pages.
        """
        key = _polygon_key()
        if not key:
            return None

        rows: list[dict] = []
        url    = "https://api.polygon.io/v2/reference/news"
        params: dict = {
            "ticker":               symbol.upper(),
            "published_utc.gte":    start,
            "published_utc.lte":    end,
            "order":                "asc",
            "limit":                1000,
            "apiKey":               key,
        }

        try:
            while url:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                for article in data.get("results", []):
                    pub      = article.get("published_utc", "")
                    insights = article.get("insights") or []
                    headline = article.get("title", "")

                    # Find insights matching this ticker (case-insensitive)
                    ticker_insights = [
                        i for i in insights
                        if str(i.get("ticker", "")).upper() == symbol.upper()
                    ]

                    if ticker_insights:
                        raw_sent = ticker_insights[0].get("sentiment", "neutral").lower()
                    else:
                        # Article mentions ticker (in tickers list) but no per-ticker insight
                        # Use the most common insight sentiment as a proxy, if any
                        if insights:
                            from collections import Counter
                            raw_sent = Counter(
                                i.get("sentiment", "neutral").lower() for i in insights
                            ).most_common(1)[0][0]
                        else:
                            raw_sent = "neutral"

                    score_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0,
                                 "bullish": 1.0, "bearish": -1.0}
                    score = score_map.get(raw_sent, 0.0)

                    rows.append({
                        "date":       pd.to_datetime(pub, utc=True).tz_convert(None).normalize(),
                        "headline":   headline,
                        "sent_score": score,
                        "weight":     1.0,
                    })

                next_url = data.get("next_url")
                if next_url:
                    # Re-append apiKey — cursor URLs drop auth params
                    url    = next_url + ("&" if "?" in next_url else "?") + f"apiKey={key}"
                    params = {}          # next_url already encodes all other params
                    logger.info("Polygon news: sleeping 12s for rate limit before next page")
                    time.sleep(self._POLYGON_RATE_LIMIT_S)
                else:
                    break

        except Exception as exc:
            logger.warning("Polygon news sentiment error for %s: %s", symbol, exc)
            return None

        return pd.DataFrame(rows) if rows else None

    def _av_news_sentiment(
        self,
        symbol: str,
        start:  str,
        end:    str,
        allow_stale: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Alpha Vantage NEWS_SENTIMENT endpoint.
        Returns ticker-specific sentiment score (-1 to +1) weighted by relevance.
        Rate limit: 25 calls/day on free tier — results are cached to disk for
        24 hours so repeated pipeline runs within a day never re-fetch.

        allow_stale : when True, return a stale (>24h) cache entry rather than
                      None when a live call is rate-limited.  This ensures
                      subsequent pipeline runs on the same day always return
                      data even after the quota is exhausted.
        """
        import pickle

        # --- Cache paths ---
        _CACHE_DIR = Path(__file__).parent / ".pipeline_cache"
        _CACHE_DIR.mkdir(exist_ok=True)
        _cache_key  = f"av_news_{symbol.upper()}_{start}_{end}.pkl"
        _cache_path = _CACHE_DIR / _cache_key
        _now        = time.time()

        # --- Cache: use any existing valid entry, never re-fetch if data is present ---
        if _cache_path.exists():
            try:
                cached = pickle.loads(_cache_path.read_bytes())
                if cached is not None:
                    logger.debug("AV cache hit: %s (%s → %s)", symbol, start, end)
                    return cached
            except Exception:
                pass  # corrupt cache — fall through to live fetch

        key = _alphavantage_key()
        if not key:
            return None

        # AV uses YYYYMMDDTHHMM format
        time_from = start.replace("-", "") + "T0000"
        time_to   = end.replace("-", "")   + "T2359"

        try:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function":  "NEWS_SENTIMENT",
                    "tickers":   symbol.upper(),
                    "time_from": time_from,
                    "time_to":   time_to,
                    "sort":      "EARLIEST",
                    "limit":     1000,
                    "apikey":    key,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()

            if "Information" in data or "Note" in data:
                # Rate-limit hit — return stale cache if available, else None
                logger.warning("AlphaVantage rate-limited for %s news sentiment", symbol)
                if allow_stale and _cache_path.exists():
                    try:
                        stale = pickle.loads(_cache_path.read_bytes())
                        if stale is not None:
                            logger.info("AV cache fallback (stale): %s (%s → %s)", symbol, start, end)
                            return stale
                    except Exception:
                        pass
                return None

            rows: list[dict] = []
            for article in data.get("feed", []):
                pub_raw = article.get("time_published", "")
                headline = article.get("title", "")
                try:
                    dt = datetime.datetime.strptime(pub_raw, "%Y%m%dT%H%M%S")
                except ValueError:
                    continue

                # Find ticker-specific sentiment (relevance-weighted)
                ticker_sents = [
                    ts for ts in article.get("ticker_sentiment", [])
                    if str(ts.get("ticker", "")).upper() == symbol.upper()
                ]
                if not ticker_sents:
                    continue

                ts        = ticker_sents[0]
                score     = float(ts.get("ticker_sentiment_score", 0.0))
                relevance = max(float(ts.get("relevance_score", 0.5)), 0.1)

                rows.append({
                    "date":       pd.Timestamp(dt).normalize(),
                    "headline":   headline,
                    "sent_score": score,
                    "weight":     relevance,
                })

            result = pd.DataFrame(rows) if rows else pd.DataFrame()
            # Cache successful result (DataFrame, possibly empty — not None)
            try:
                _cache_path.write_bytes(pickle.dumps(result))
                logger.debug("AV cache written: %s (%s → %s) %d rows", symbol, start, end, len(result))
            except Exception:
                pass
            return result if not result.empty else None

        except Exception as exc:
            logger.warning("AlphaVantage news sentiment error for %s: %s", symbol, exc)
            return None

    def _alpaca_news_sentiment(
        self,
        symbol: str,
        start:  str,
        end:    str,
    ) -> Optional[pd.DataFrame]:
        """
        Alpaca REST news API (data.alpaca.markets/v1beta1/news).
        Returns per-ticker AI ``insights`` with sentiment + sentiment_reasoning,
        plus keywords — this is the 'Massive'-style structured sentiment data.

        Auth: uses the Alpaca API key/secret stored in keys.txt under "massive"
        or standard ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables.

        Pagination handled via ``next_page_token``.
        Rate limit: ~200 req/min on free tier.
        """
        # Try the 'massive' key as the Alpaca secret, std key ID as header
        api_key    = _alpaca_key() or _get_key("ALPACA_API_KEY", "alpaca code")
        secret_key = _get_key("ALPACA_SECRET_KEY", "massive")   # massive label = Alpaca secret
        if not api_key and not secret_key:
            return None

        rows: list[dict] = []
        url     = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        params: dict = {
            "symbols":         symbol.upper(),
            "start":           start,
            "end":             end,
            "limit":           50,           # max per page for REST news
            "include_content": "false",
            "sort":            "asc",
        }

        try:
            while True:
                resp = requests.get(url, headers=headers, params=params, timeout=15)
                if resp.status_code in (401, 403):
                    logger.warning("Alpaca news sentiment: unauthorised (key issue) for %s", symbol)
                    break
                resp.raise_for_status()
                data = resp.json()

                for article in data.get("news", []):
                    pub      = article.get("created_at", "")
                    headline = article.get("headline", "")
                    insights = article.get("insights") or []

                    ticker_insights = [
                        i for i in insights
                        if str(i.get("ticker", "")).upper() == symbol.upper()
                    ]

                    if ticker_insights:
                        raw_sent = ticker_insights[0].get("sentiment", "neutral").lower()
                    elif insights:
                        from collections import Counter
                        raw_sent = Counter(
                            i.get("sentiment", "neutral").lower() for i in insights
                        ).most_common(1)[0][0]
                    else:
                        raw_sent = "neutral"

                    score_map = {
                        "positive": 1.0, "negative": -1.0, "neutral": 0.0,
                        "bullish":  1.0, "bearish":  -1.0,
                    }
                    score = score_map.get(raw_sent, 0.0)

                    rows.append({
                        "date":       pd.to_datetime(pub).tz_localize(None).normalize()
                                      if pd.to_datetime(pub).tzinfo is None
                                      else pd.to_datetime(pub).tz_convert(None).normalize(),
                        "headline":   headline,
                        "sent_score": score,
                        "weight":     1.0,
                    })

                token = data.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token

        except Exception as exc:
            logger.warning("Alpaca news sentiment error for %s: %s", symbol, exc)
            return None

        return pd.DataFrame(rows) if rows else None

    # -----------------------------------------------------------------------
    # SENTIMENT (insider transactions)
    # -----------------------------------------------------------------------

    def sentiment(
        self,
        symbol: str,
        start:  str = "",
        end:    str = "",
    ) -> pd.DataFrame:
        """
        Monthly insider sentiment (net buy/sell).  Returns DataFrame with:
        year, month, change (net insider shares), mspr (buy ratio 0-1)

        Primary  : yfinance insider_transactions (free, ~12 months of history)
        Fallback : Finnhub insider_sentiment     (requires premium plan)
        """
        df = self._yf_insider_sentiment(symbol)
        if df is not None and not df.empty:
            return df

        return self._finnhub_insider_sentiment(symbol, start, end)

    def _yf_insider_sentiment(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Build monthly insider sentiment from yfinance insider_transactions.
        Columns: year, month, change (net shares bought), mspr (buy_vol / total_vol).
        """
        try:
            t   = yf.Ticker(symbol)
            txn = t.insider_transactions
            if txn is None or txn.empty:
                return None

            df = txn.copy()

            # Parse date
            date_col = "Start Date" if "Start Date" in df.columns else df.columns[0]
            df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
            df = df.dropna(subset=["date"])

            # Parse transaction type from text
            txt = df["Text"].fillna("").str.lower() if "Text" in df.columns else pd.Series("", index=df.index)
            is_buy  = txt.str.contains(r"purchase|award|grant", regex=True)
            is_sell = txt.str.contains(r"sale|sell",            regex=True)

            shares = pd.to_numeric(df["Shares"], errors="coerce").fillna(0)

            df["bought"] = shares.where(is_buy,  0)
            df["sold"]   = shares.where(is_sell, 0)

            df["year"]  = df["date"].dt.year
            df["month"] = df["date"].dt.month

            monthly = (
                df.groupby(["year", "month"])
                .agg(bought=("bought", "sum"), sold=("sold", "sum"))
                .reset_index()
            )
            monthly["change"] = monthly["bought"] - monthly["sold"]
            total = monthly["bought"] + monthly["sold"]
            monthly["mspr"] = (monthly["bought"] / total.replace(0, float("nan"))).round(4)

            return monthly[["year", "month", "change", "mspr"]]

        except Exception as exc:
            logger.warning("yfinance insider sentiment error for %s: %s", symbol, exc)
            return None

    def _finnhub_insider_sentiment(
        self, symbol: str, start: str = "", end: str = ""
    ) -> pd.DataFrame:
        key = _finnhub_key()
        if not key:
            return pd.DataFrame()
        if not start:
            start = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
        if not end:
            end = datetime.date.today().isoformat()
        url = "https://finnhub.io/api/v1/stock/insider-sentiment"
        try:
            resp = requests.get(
                url, params={"symbol": symbol, "_from": start, "to": end, "token": key},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data") or []
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)
        except Exception as exc:
            logger.warning("Finnhub sentiment error for %s: %s", symbol, exc)
            return pd.DataFrame()

    # -----------------------------------------------------------------------
    # TECHNICALS (computed indicators via yfinance + pandas)
    # -----------------------------------------------------------------------

    def technicals(
        self,
        symbol:   str,
        period:   str = "1y",
        interval: str = "1d",
        start:    str = "",
        end:      str = "",
    ) -> pd.DataFrame:
        """
        OHLCV bars enriched with common technical indicators.
        Computed using pandas on top of yfinance data (no extra API calls).

        Indicators added:
        SMA_20, SMA_50, SMA_200, EMA_20,
        RSI_14, MACD, MACD_signal, MACD_hist,
        BB_upper, BB_mid, BB_lower,
        ATR_14, OBV, VWAP (intraday), daily_return, log_return

        Parameters
        ----------
        start / end : 'YYYY-MM-DD' — if both supplied, used instead of period.
                      Pass a start that is earlier than you need to warm up
                      rolling indicators; the caller trims back afterward.
        """
        df = self._yf_ohlcv(symbol, start, end, period, interval)
        if df is None or df.empty:
            return pd.DataFrame()

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        # Moving averages
        df["SMA_20"]  = close.rolling(20).mean()
        df["SMA_50"]  = close.rolling(50).mean()
        df["SMA_200"] = close.rolling(200).mean()
        df["EMA_20"]  = close.ewm(span=20, adjust=False).mean()

        # RSI-14
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"]         = ema12 - ema26
        df["MACD_signal"]  = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"]    = df["MACD"] - df["MACD_signal"]

        # Bollinger Bands (20, 2σ)
        df["BB_mid"]   = close.rolling(20).mean()
        bb_std         = close.rolling(20).std()
        df["BB_upper"] = df["BB_mid"] + 2 * bb_std
        df["BB_lower"] = df["BB_mid"] - 2 * bb_std

        # ATR-14
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        df["ATR_14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        # OBV
        direction  = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        df["OBV"]  = (direction * volume).cumsum()

        # Returns
        df["daily_return"] = close.pct_change()
        df["log_return"]   = (close / close.shift()).apply(
            lambda x: float("nan") if x <= 0 else __import__("math").log(x)
        )

        return df

    # -----------------------------------------------------------------------
    # OPTIONS (options chain from yfinance; enriched with Polygon greeks)
    # -----------------------------------------------------------------------

    def options(
        self,
        symbol:          str,
        expiration_date: Optional[str] = None,   # 'YYYY-MM-DD'; defaults to nearest expiry
        contract_type:   str           = "both",  # 'call' | 'put' | 'both'
        enrich_polygon:  bool          = True,
    ) -> pd.DataFrame:
        """
        Options chain.  Returns DataFrame with standard option fields:
        contractSymbol, strike, lastPrice, bid, ask, mid, impliedVolatility,
        openInterest, volume, inTheMoney, expiration, optionType

        Primary  : yfinance (free, basic surface — no live greeks on free)
        Enriched : Polygon  (live greeks delta/gamma/theta/vega + IV — Starter+ plan)
        """
        df = self._yf_options(symbol, expiration_date, contract_type)
        if df is None or df.empty:
            return pd.DataFrame()

        if enrich_polygon and _polygon_key():
            df = self._enrich_with_polygon_greeks(df, symbol, expiration_date)

        return df

    def _yf_options(self, symbol, expiration_date, contract_type) -> Optional[pd.DataFrame]:
        try:
            t = yf.Ticker(symbol)
            expiries = t.options
            if not expiries:
                return None
            if expiration_date:
                exp = expiration_date
                if exp not in expiries:
                    # pick nearest available
                    exp = min(expiries, key=lambda d: abs(
                        (datetime.date.fromisoformat(d) - datetime.date.fromisoformat(expiration_date)).days
                    ))
            else:
                today = datetime.date.today()
                # Skip expiries that are today or already past; prefer ≥7 days out
                future = [e for e in expiries
                          if datetime.date.fromisoformat(e) > today]
                week_plus = [e for e in future
                             if (datetime.date.fromisoformat(e) - today).days >= 7]
                exp = week_plus[0] if week_plus else (future[0] if future else expiries[0])

            chain = t.option_chain(exp)
            frames = []
            if contract_type in ("call", "both"):
                calls = chain.calls.copy()
                calls["optionType"] = "call"
                frames.append(calls)
            if contract_type in ("put", "both"):
                puts = chain.puts.copy()
                puts["optionType"] = "put"
                frames.append(puts)
            df = pd.concat(frames, ignore_index=True)
            df["expiration"] = exp
            df["mid"] = (df["bid"] + df["ask"]) / 2
            return df
        except Exception as exc:
            logger.warning("yfinance options error for %s: %s", symbol, exc)
            return None

    def _enrich_with_polygon_greeks(
        self, df: pd.DataFrame, symbol: str, expiration_date: Optional[str]
    ) -> pd.DataFrame:
        """
        Fetch Polygon snapshot for options and merge delta/gamma/theta/vega/IV.
        Only available on Starter+ plan; gracefully returns df unchanged on error.
        Rate limit: 5 req/min free tier.
        """
        key = _polygon_key()
        if not key:
            return df
        url  = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper()}"
        params: dict = {"apiKey": key, "limit": 250, "order": "asc", "sort": "expiration_date"}
        if expiration_date:
            params["expiration_date"] = expiration_date
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if not results:
                return df
            greek_rows = []
            for r in results:
                det  = r.get("details") or {}
                grk  = r.get("greeks") or {}
                day  = r.get("day") or {}
                greek_rows.append({
                    "contractSymbol": det.get("ticker", ""),
                    "polygon_delta":  grk.get("delta"),
                    "polygon_gamma":  grk.get("gamma"),
                    "polygon_theta":  grk.get("theta"),
                    "polygon_vega":   grk.get("vega"),
                    "polygon_iv":     r.get("implied_volatility"),
                    "polygon_oi":     r.get("open_interest"),
                })
            greeks_df = pd.DataFrame(greek_rows).drop_duplicates("contractSymbol")
            if "contractSymbol" in df.columns:
                df = df.merge(greeks_df, on="contractSymbol", how="left")
        except Exception as exc:
            logger.warning("Polygon greeks enrichment error for %s: %s", symbol, exc)
        return df

    # -----------------------------------------------------------------------
    # FILINGS (Fiscal.ai — structured SEC filing data)
    # -----------------------------------------------------------------------

    def filings(
        self,
        symbol:    str,
        form_type: str = "10-K",
        limit:     int = 5,
    ) -> pd.DataFrame:
        """
        Parsed SEC filings from Fiscal.ai.

        Returns DataFrame — schema depends on Fiscal.ai response.
        TODO: column mapping must be updated once Fiscal.ai API schema confirmed.

        Parameters
        ----------
        form_type : '10-K' | '10-Q' | '8-K' | 'DEF 14A'
        limit     : number of filings to return
        """
        key = _fiscal_ai_key()
        if not key:
            logger.warning("No FISCAL_AI_API_KEY — filings unavailable")
            return pd.DataFrame()
        company_id = _fiscal_ai_company_id(symbol)
        if not company_id:
            logger.debug("Fiscal.ai filings: %s not in free-tier company list — skipping", symbol)
            return pd.DataFrame()
        url = f"https://api.fiscal.ai/v1/company/{company_id}/filings"
        headers = {"Authorization": f"Bearer {key}"}
        try:
            resp = requests.get(
                url,
                headers=headers,
                params={"form_type": form_type, "limit": limit},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json().get("filings") or []
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)
        except Exception as exc:
            logger.warning("Fiscal.ai filings error for %s: %s", symbol, exc)
            return pd.DataFrame()

    # -----------------------------------------------------------------------
    # CALENDAR (trading days + earnings dates)
    # -----------------------------------------------------------------------

    def calendar(
        self,
        start: str = "",
        end:   str = "",
    ) -> pd.DataFrame:
        """
        Market calendar: trading days + holidays.
        Source: Alpaca market calendar (falls back to pandas_market_calendars if Alpaca unavailable).

        Returns DataFrame with columns: date, open, close, session_open, session_close.
        """
        if not start:
            start = datetime.date.today().isoformat()
        if not end:
            end = (datetime.date.today() + datetime.timedelta(days=90)).isoformat()

        df = self._alpaca_calendar(start, end)
        if df is not None and not df.empty:
            return df

        logger.warning("Alpaca calendar unavailable — generating from pandas_market_calendars")
        return self._fallback_calendar(start, end)

    def _alpaca_calendar(self, start, end) -> Optional[pd.DataFrame]:
        _, _, trade_client = self._alpaca_clients()
        if trade_client is None:
            return None
        try:
            import datetime as _dt
            req = GetCalendarRequest(
                start=_dt.date.fromisoformat(start),
                end=_dt.date.fromisoformat(end),
            )
            sessions = trade_client.get_calendar(req)
            if not sessions:
                return None
            rows = [{
                "date":  s.date,
                "open":  str(s.open),
                "close": str(s.close),
            } for s in sessions]
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as exc:
            logger.warning("Alpaca SDK calendar error: %s", exc)
            return None

    def _fallback_calendar(self, start, end) -> pd.DataFrame:
        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar("NYSE")
            sched = nyse.schedule(start_date=start, end_date=end)
            df = sched.reset_index()
            df.columns = ["date", "open", "close"]
            return df
        except ImportError:
            # Pure pandas fallback: business days only (approximate)
            dates = pd.bdate_range(start=start, end=end)
            return pd.DataFrame({"date": dates})
        except Exception as exc:
            logger.warning("Fallback calendar error: %s", exc)
            return pd.DataFrame()

    def earnings_calendar(self, symbol: str, limit: int = 8) -> pd.DataFrame:
        """
        Earnings dates + EPS estimate + actual for a specific symbol.
        Source: Finnhub earnings_calendar → yfinance earnings_dates fallback.
        """
        key = _finnhub_key()
        today = datetime.date.today()
        start = (today - datetime.timedelta(days=365 * 2)).isoformat()
        end   = (today + datetime.timedelta(days=365)).isoformat()

        if key:
            url = "https://finnhub.io/api/v1/calendar/earnings"
            try:
                resp = requests.get(
                    url,
                    params={"symbol": symbol, "_from": start, "to": end, "token": key},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json().get("earningsCalendar") or []
                if data:
                    return pd.DataFrame(data[:limit])
            except Exception as exc:
                logger.warning("Finnhub earnings calendar error for %s: %s", symbol, exc)

        # yfinance fallback
        try:
            df = yf.Ticker(symbol).earnings_dates
            if df is not None and not df.empty:
                return df.head(limit).reset_index()
        except Exception as exc:
            logger.warning("yfinance earnings_dates error for %s: %s", symbol, exc)

        return pd.DataFrame()

    # -----------------------------------------------------------------------
    # API CONNECTIVITY TEST
    # -----------------------------------------------------------------------

    def test_apis(self, symbol: str = "AAPL") -> pd.DataFrame:
        """
        Test each data source for live connectivity.
        Returns a DataFrame summarising: provider, method, status, detail, latency_ms

        Usage:
            ex = DataExtractor()
            results = ex.test_apis("AAPL")
            print(results.to_string(index=False))
        """
        results = []

        def _run(provider, method_name, call):
            t0 = time.perf_counter()
            try:
                data = call()
                ok   = data is not None and (
                    (isinstance(data, pd.DataFrame) and not data.empty) or
                    (isinstance(data, (dict, list)) and len(data) > 0)
                )
                latency = int((time.perf_counter() - t0) * 1000)
                status  = "OK" if ok else "EMPTY"
                detail  = f"{len(data)} rows" if isinstance(data, pd.DataFrame) else "non-empty"
                if not ok:
                    detail = "returned empty result"
            except Exception as exc:
                latency = int((time.perf_counter() - t0) * 1000)
                status  = "FAIL"
                detail  = str(exc)[:120]
            results.append({
                "provider":   provider,
                "method":     method_name,
                "status":     status,
                "detail":     detail,
                "latency_ms": latency,
            })

        today = datetime.date.today().isoformat()
        month_ago = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()

        # yfinance — backbone, should always pass
        _run("yfinance", "ohlcv",         lambda: self._yf_ohlcv(symbol, "", "", "1mo", "1d"))
        _run("yfinance", "profile",       lambda: pd.DataFrame([self._yf_profile(symbol)]))
        _run("yfinance", "fundamentals",  lambda: pd.DataFrame([self._yf_fundamentals(symbol)]))
        _run("yfinance", "news",          lambda: self._yf_news(symbol, 10))
        _run("yfinance", "options",       lambda: self._yf_options(symbol, None, "call"))
        _run("yfinance", "technicals",    lambda: self.technicals(symbol, period="3mo"))

        # Finnhub
        _run("finnhub",  "profile",       lambda: pd.DataFrame([self._finnhub_profile(symbol) or {}]))
        _run("finnhub",  "fundamentals",  lambda: pd.DataFrame([self._finnhub_fundamentals(symbol) or {}]))
        _run("finnhub",  "news",          lambda: self._finnhub_news(symbol, month_ago, today, 10))
        _run("finnhub",  "sentiment",     lambda: self.sentiment(symbol, month_ago, today))
        _run("finnhub",  "earnings_cal",  lambda: self.earnings_calendar(symbol))

        # Fiscal.ai (stub — will FAIL until endpoint confirmed)
        _run("fiscal_ai","fundamentals",  lambda: self._fiscal_ai_fundamentals(symbol) or {})
        _run("fiscal_ai","filings",       lambda: self.filings(symbol, "10-K", 2))

        # Polygon OHLCV — works on free tier (SIP tape, 15-min delayed)
        _run("polygon",  "ohlcv",         lambda: self._polygon_ohlcv_raw(symbol, month_ago, today))
        # Polygon options snapshot — requires Starter+ plan; 403 on free key
        _run("polygon",  "options_chain", lambda: pd.DataFrame(self._polygon_options_raw(symbol)))

        # Alpaca (requires ALPACA_SECRET_KEY in env — not in keys.txt)
        _run("alpaca",   "ohlcv",         lambda: self._alpaca_ohlcv(symbol, month_ago, today, "1d"))
        _run("alpaca",   "news",          lambda: self._alpaca_news(symbol, month_ago, today, 10))
        _run("alpaca",   "calendar",      lambda: self._alpaca_calendar(month_ago, today))

        df = pd.DataFrame(results)
        _ok    = (df["status"] == "OK").sum()
        _empty = (df["status"] == "EMPTY").sum()
        _fail  = (df["status"] == "FAIL").sum()
        print(f"\n{'─'*70}")
        print(f"  API Test Results for {symbol}")
        print(f"{'─'*70}")
        print(df.to_string(index=False))
        print(f"{'─'*70}")
        print(f"  PASS: {_ok}   EMPTY: {_empty}   FAIL: {_fail}   Total: {len(df)}")
        print(f"{'─'*70}\n")
        return df

    def _polygon_ohlcv_raw(self, symbol, start, end) -> pd.DataFrame:
        """Polygon SIP-tape OHLCV — works on free tier (15-min delayed)."""
        key = _polygon_key()
        if not key:
            return pd.DataFrame()
        import datetime as _dt
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start}/{end}"
        resp = requests.get(url, params={"apiKey": key, "adjusted": "true",
                                         "sort": "asc", "limit": 50000}, timeout=15)
        resp.raise_for_status()
        bars = resp.json().get("results") or []
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                                 "c": "Close", "v": "Volume", "vw": "VWAP"})
        return df.set_index("Date")[["Open", "High", "Low", "Close", "Volume", "VWAP"]]

    def _polygon_options_raw(self, symbol) -> list:
        """Raw Polygon options call used by test_apis(). Requires Starter+ plan."""
        key = _polygon_key()
        if not key:
            return []
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper()}"
        try:
            resp = requests.get(url, params={"apiKey": key, "limit": 10}, timeout=15)
            if resp.status_code == 403:
                logger.warning("Polygon options snapshot requires Starter+ plan (403). "
                               "OHLCV aggs (free tier) still works.")
                return []
            resp.raise_for_status()
            return resp.json().get("results") or []
        except Exception as exc:
            logger.warning("Polygon options error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# CLI entry point — python extractor.py [SYMBOL]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    ex = DataExtractor()

    print(f"\n=== OHLCV (last 5 rows) ===")
    bars = ex.ohlcv(symbol, period="1mo")
    print(bars.tail(5).to_string())

    print(f"\n=== PROFILE ===")
    print(ex.profile(symbol).to_string(index=False))

    print(f"\n=== FUNDAMENTALS (key ratios) ===")
    funds = ex.fundamentals(symbol)
    key_cols = ["pe_ratio", "forward_pe", "pb_ratio", "roe", "net_margin",
                "ev_to_ebitda", "debt_to_equity", "revenue_growth", "beta"]
    print(funds[[c for c in key_cols if c in funds.columns]].to_string(index=False))

    print(f"\n=== API TEST ===")
    ex.test_apis(symbol)
