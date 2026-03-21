"""
ARIMA Price Forecasting Model

Auto-Regressive Integrated Moving Average (ARIMA) for financial time series.

Model:  ARIMA(p, d, q)
  p — AR order: uses past p prices / returns
  d — differencing order (1 = log-return differencing for stationarity)
  q — MA order: uses past q forecast errors

Order selection via AIC / BIC grid-search over (p, d, q) combinations.
Includes Ljung-Box residual test: p-value > 0.05 → white-noise residuals (good).

Requires: statsmodels  (pip install statsmodels)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import product
from typing import List
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ARIMAOrder:
    p: int
    d: int
    q: int

    def __str__(self) -> str:
        return f"ARIMA({self.p},{self.d},{self.q})"


@dataclass
class ARIMAForecast:
    order: ARIMAOrder
    aic: float
    bic: float
    in_sample_rmse: float
    ljung_box_p_value: float          # > 0.05 = white-noise residuals
    forecast: List[float] = field(default_factory=list)
    confidence_lower: List[float] = field(default_factory=list)
    confidence_upper: List[float] = field(default_factory=list)


class ARIMAModel:
    """
    ARIMA forecaster with automatic order selection.

    Parameters
    ----------
    order     : ARIMAOrder, optional — if None, auto-selects via AIC grid search.
    max_p     : int — maximum AR order to consider during grid search.
    max_q     : int — maximum MA order to consider during grid search.
    """

    def __init__(self, order: ARIMAOrder = None, max_p: int = 4, max_q: int = 2):
        self.order = order
        self.max_p = max_p
        self.max_q = max_q

    # ------------------------------------------------------------------
    # Order selection
    # ------------------------------------------------------------------

    def auto_select_order(self, series: pd.Series, d: int = 1) -> ARIMAOrder:
        """Grid-search (p, q) over [0..max_p] x [0..max_q], minimising AIC."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            return ARIMAOrder(p=1, d=d, q=0)

        best_aic = np.inf
        best_order = ARIMAOrder(1, d, 0)

        for p, q in product(range(self.max_p + 1), range(self.max_q + 1)):
            if p == 0 and q == 0:
                continue
            try:
                res = ARIMA(series, order=(p, d, q)).fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = ARIMAOrder(p=p, d=d, q=q)
            except Exception:
                pass

        return best_order

    # ------------------------------------------------------------------
    # Fit & forecast
    # ------------------------------------------------------------------

    def fit_and_forecast(
        self,
        series: pd.Series,
        steps: int = 5,
        d: int = 1,
        confidence: float = 0.95,
    ) -> ARIMAForecast:
        """Fit ARIMA and return out-of-sample forecast with confidence intervals."""
        order = self.order or self.auto_select_order(series, d=d)

        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.stats.diagnostic import acorr_ljungbox

            model  = ARIMA(series, order=(order.p, order.d, order.q))
            result = model.fit()

            resid = result.resid.dropna()
            rmse  = float(np.sqrt((resid ** 2).mean()))

            lb    = acorr_ljungbox(resid, lags=[10], return_df=True)
            lb_p  = float(lb['lb_pvalue'].iloc[0])

            fc    = result.get_forecast(steps=steps)
            ci    = fc.conf_int(alpha=1 - confidence)

            return ARIMAForecast(
                order=order,
                aic=float(result.aic),
                bic=float(result.bic),
                in_sample_rmse=rmse,
                ljung_box_p_value=lb_p,
                forecast=fc.predicted_mean.tolist(),
                confidence_lower=ci.iloc[:, 0].tolist(),
                confidence_upper=ci.iloc[:, 1].tolist(),
            )

        except ImportError:
            # Fallback: simple differenced AR(1)
            diffs = series.diff(order.d).dropna().values
            phi   = np.cov(diffs[1:], diffs[:-1])[0, 1] / np.var(diffs[:-1])
            sigma = float(np.std(diffs - phi * np.roll(diffs, 1)))
            last  = diffs[-1]
            fc_vals = [float(last * phi ** i) for i in range(1, steps + 1)]
            margin  = 1.96 * sigma
            return ARIMAForecast(
                order=order, aic=np.nan, bic=np.nan,
                in_sample_rmse=sigma, ljung_box_p_value=np.nan,
                forecast=fc_vals,
                confidence_lower=[v - margin for v in fc_vals],
                confidence_upper=[v + margin for v in fc_vals],
            )

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def run(self, ticker: str, period: str = '2y', forecast_days: int = 5) -> None:
        import yfinance as yf
        data  = yf.download(ticker, period=period, progress=False)
        close = data['Close'].squeeze()

        result = self.fit_and_forecast(close, steps=forecast_days)
        last_price = float(close.iloc[-1])

        print(f"\n{ticker}  {result.order}  [{period}]")
        print(f"  AIC: {result.aic:.1f}   BIC: {result.bic:.1f}")
        print(f"  In-sample RMSE: {result.in_sample_rmse:.4f}")
        lb = result.ljung_box_p_value
        status = 'white noise ✓' if (not np.isnan(lb) and lb > 0.05) else 'autocorrelation ✗'
        print(f"  Ljung-Box p-value: {lb:.4f}  ({status})")
        print(f"\n  {forecast_days}-day forecast  (last price ${last_price:.2f}):")
        for i, (fc, lo, hi) in enumerate(
            zip(result.forecast, result.confidence_lower, result.confidence_upper), 1
        ):
            print(f"    Day {i}: ${fc:.2f}  [{lo:.2f}, {hi:.2f}]")


if __name__ == '__main__':
    ARIMAModel().run('NVDA')
