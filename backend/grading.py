"""
grading.py — fundamental grades for the Metrics & Grades tab.

Pure functions over Finnhub's basic-financials `metric` dict. A missing input
leaves its category ungraded (N/A): an earlier version silently substituted
"average" values, which graded ETFs and unknown tickers C and gave loss-makers
a P/E of 20. Percent inputs are clamped to ±50 so one outlier (an ROE of 190%
off a tiny equity base, a one-off revenue jump) can't dominate an average.
Thresholds are market-wide, not sector-relative.
"""
import math

_CLAMP = 50.0
_SCORES = {'A': 95, 'B': 85, 'C': 75, 'D': 60, 'F': 40}


def _num(metric, key):
    try:
        v = float(metric.get(key))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _grade(x, cuts, higher_is_better):
    """A–F from four cut points ordered best grade first."""
    if higher_is_better:
        idx = sum(x <= c for c in cuts)          # above every cut → A
    else:
        idx = sum(x >= c for c in cuts)          # below every cut → A
    return 'ABCDF'[idx]


def _category(grade, description, inputs):
    return {'grade': grade, 'score': _SCORES.get(grade), 'description': description,
            'inputs': {k: v for k, v in inputs.items() if v is not None}}


def _mean_clamped(*vals):
    vals = [max(-_CLAMP, min(_CLAMP, v)) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def grade_fundamentals(metric: dict) -> dict:
    """Per-category grades plus an overall grade over the categories with data."""
    pe = _num(metric, 'peBasicExclExtraTTM')
    roe = _num(metric, 'roeTTM')
    margin = _num(metric, 'netProfitMarginTTM')
    eps_g = _num(metric, 'epsGrowthTTMYoy')
    rev_g = _num(metric, 'revenueGrowthTTMYoy')
    de = _num(metric, 'totalDebt/totalEquityQuarterly')

    if pe is None:
        valuation = _category(None, 'No P/E available', {})
    elif pe <= 0:
        valuation = _category(None, 'P/E not meaningful: the company has negative earnings',
                              {'P/E (TTM)': pe})
    else:
        g = _grade(pe, [15, 20, 25, 35], higher_is_better=False)
        valuation = _category(g, {
            'A': 'Low P/E relative to the broad market', 'B': 'P/E near the market average',
            'C': 'P/E slightly above the market average', 'D': 'P/E well above the market average',
            'F': 'Very high P/E'}[g], {'P/E (TTM)': pe})

    prof = _mean_clamped(roe, margin)
    profitability = _category(None, 'No profitability data', {}) if prof is None else _category(
        _grade(prof, [20, 15, 10, 5], higher_is_better=True),
        f'Average of ROE and net margin: {prof:.1f}%',
        {'ROE % (TTM)': roe, 'Net margin % (TTM)': margin})

    growth_v = _mean_clamped(eps_g, rev_g)
    growth = _category(None, 'No growth data', {}) if growth_v is None else _category(
        _grade(growth_v, [20, 15, 10, 5], higher_is_better=True),
        f'Average of EPS and revenue growth: {growth_v:.1f}% year over year',
        {'EPS growth % (YoY)': eps_g, 'Revenue growth % (YoY)': rev_g})

    if de is None:
        health = _category(None, 'No debt/equity data', {})
    elif de < 0:
        health = _category(None, 'Debt/equity not meaningful: shareholder equity is negative',
                           {'Debt/Equity': de})
    else:
        g = _grade(de, [0.3, 0.5, 1.0, 2.0], higher_is_better=False)
        health = _category(g, {
            'A': 'Low debt', 'B': 'Modest debt', 'C': 'Moderate debt',
            'D': 'High debt', 'F': 'Very high debt (normal for banks, which this '
                                   'market-wide scale does not adjust for)'}[g],
                           {'Debt/Equity': de})

    cats = {'valuation': valuation, 'profitability': profitability,
            'growth': growth, 'financial_health': health}
    scores = [c['score'] for c in cats.values() if c['score'] is not None]
    avg = sum(scores) / len(scores) if scores else None
    overall = None if avg is None else (
        'A' if avg >= 90 else 'B' if avg >= 80 else 'C' if avg >= 70 else 'D' if avg >= 60 else 'F')
    return {'overall_grade': overall, 'average_score': avg,
            'graded_categories': len(scores), 'metrics': cats}
