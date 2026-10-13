"""
grading.py — fundamental grades for the Metrics & Grades tab.

Pure functions over Finnhub basic-financials `metric` dicts: the company's and
its industry peers'. Each input is graded by the company's percentile among the
peers that report it, so the bar is set by the industry (a bank's P/B is only
compared with other banks') and an outlier can move a rank by one place, never
drag an average. With fewer than MIN_PEERS peers reporting an input, the six
inputs the original market-wide scale covered fall back to it; the rest stay
ungraded. A missing input is never filled in: an earlier version substituted
"average" values, which graded ETFs and unknown tickers C.
"""
import math

MIN_PEERS = 4
# Finnhub profile `finnhubIndustry` values
FINANCIAL_INDUSTRIES = {'Banking', 'Insurance', 'Financial Services'}

# category → [(label, Finnhub key, higher_is_better, market-wide cut points best
#              grade first or None, why a negative value is meaningless or None)]
_INPUTS = {
    'valuation': [
        ('P/E (TTM)', 'peBasicExclExtraTTM', False, (15, 20, 25, 35), 'negative earnings'),
        # so a fast grower's P/E isn't read as plain expensive
        ('PEG (TTM)', 'pegTTM', False, None, 'negative earnings or shrinking EPS'),
        ('P/S (TTM)', 'psTTM', False, None, None),
        ('P/B', 'pbQuarterly', False, None, 'negative equity'),
        ('EV/EBITDA (TTM)', 'evEbitdaTTM', False, None, 'negative EBITDA'),
    ],
    'profitability': [
        ('ROE % (TTM)', 'roeTTM', True, (20, 15, 10, 5), None),
        ('ROA % (TTM)', 'roaTTM', True, None, None),
        ('Net margin % (TTM)', 'netProfitMarginTTM', True, (20, 15, 10, 5), None),
        ('Operating margin % (TTM)', 'operatingMarginTTM', True, None, None),
        ('Gross margin % (TTM)', 'grossMarginTTM', True, None, None),
    ],
    'growth': [
        ('EPS growth % (YoY)', 'epsGrowthTTMYoy', True, (20, 15, 10, 5), None),
        ('Revenue growth % (YoY)', 'revenueGrowthTTMYoy', True, (20, 15, 10, 5), None),
        ('EPS growth % (3Y)', 'epsGrowth3Y', True, None, None),
        ('Revenue growth % (3Y)', 'revenueGrowth3Y', True, None, None),
    ],
    'financial_health': [
        ('Debt/Equity', 'totalDebt/totalEquityQuarterly', False, (0.3, 0.5, 1.0, 2.0), 'negative equity'),
        ('Current ratio', 'currentRatioQuarterly', True, None, None),
        ('Interest coverage (TTM)', 'netInterestCoverageTTM', True, None, None),
    ],
}

# A market-scale grade scores at the middle of the matching percentile band.
_BAND_MID = {'A': 90, 'B': 70, 'C': 50, 'D': 30, 'F': 10}


def _num(metric, key):
    try:
        v = float(metric.get(key))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _band(score):
    return 'A' if score >= 80 else 'B' if score >= 60 else 'C' if score >= 40 else 'D' if score >= 20 else 'F'


def _market_grade(x, cuts, higher_is_better):
    """A–F from four cut points ordered best grade first."""
    if higher_is_better:
        idx = sum(x <= c for c in cuts)          # above every cut → A
    else:
        idx = sum(x >= c for c in cuts)          # below every cut → A
    return 'ABCDF'[idx]


def peer_percentile(value, peer_values, higher_is_better):
    """Share of peers the company beats (0–100), ties counting half."""
    beaten = sum((v < value) if higher_is_better else (v > value) for v in peer_values)
    ties = sum(v == value for v in peer_values)
    return 100 * (beaten + 0.5 * ties) / len(peer_values)


def _grade_input(value, peer_values, higher_is_better, cuts, negative_means):
    """(input record, 0–100 score or None)."""
    rec = {'value': value, 'percentile': None, 'grade': None, 'note': None}
    if negative_means and value < 0:
        rec['note'] = f'not meaningful: {negative_means}'
        return rec, None
    if negative_means:
        # a loss-maker's negative P/E is not "cheap"
        peer_values = [v for v in peer_values if v >= 0]
    if len(peer_values) >= MIN_PEERS:
        pct = peer_percentile(value, peer_values, higher_is_better)
        rec.update(percentile=round(pct), grade=_band(pct))
        return rec, pct
    if cuts:
        rec.update(grade=_market_grade(value, cuts, higher_is_better), note='market-wide scale')
        return rec, _BAND_MID[rec['grade']]
    rec['note'] = 'too few peers'
    return rec, None


def _category(name, metric, peer_metrics):
    inputs, scores, market, ungraded, meaningless = {}, [], [], [], []
    for label, key, hib, cuts, negative_means in _INPUTS[name]:
        value = _num(metric, key)
        if value is None:
            continue
        peer_values = [v for v in (_num(m, key) for m in peer_metrics) if v is not None]
        rec, score = _grade_input(value, peer_values, hib, cuts, negative_means)
        inputs[label] = rec
        if score is not None:
            scores.append(score)
        if rec['note'] == 'market-wide scale':
            market.append(label)
        elif rec['note'] == 'too few peers':
            ungraded.append(label)
        elif rec['note']:
            meaningless.append(f"{label} {rec['note']}")

    if not inputs:
        return {'grade': None, 'score': None, 'description': 'No data', 'inputs': {}}
    score = sum(scores) / len(scores) if scores else None
    peer_ranked = len(scores) - len(market)
    parts = []
    if peer_ranked:
        parts.append(f'Average peer percentile over {peer_ranked} metric{"s" * (peer_ranked > 1)}')
    if market:
        parts.append(f'{", ".join(market)} on the market-wide scale '
                     f'(fewer than {MIN_PEERS} peers report {"it" if len(market) == 1 else "them"})')
    if ungraded:
        parts.append(f'Not graded, fewer than {MIN_PEERS} peers report: {", ".join(ungraded)}')
    parts += meaningless
    return {'grade': None if score is None else _band(score),
            'score': None if score is None else round(score),
            'description': '. '.join(parts), 'inputs': inputs}


def grade_fundamentals(metric: dict, peer_metrics=(), industry=None) -> dict:
    """Per-category grades plus an overall grade over the categories with data.

    peer_metrics: the industry peers' basic-financials `metric` dicts.
    industry: Finnhub `finnhubIndustry`, used to spot financials.
    """
    cats = {name: _category(name, metric, peer_metrics) for name in _INPUTS}
    if industry in FINANCIAL_INDUSTRIES:
        cats['financial_health'] = {
            'grade': None, 'score': None, 'inputs': {},
            'description': 'Not graded for banks, insurers and other financials: debt/equity, '
                           'current ratio and interest coverage mean little for balance sheets '
                           'funded by deposits and premiums'}
    scores = [c['score'] for c in cats.values() if c['score'] is not None]
    avg = sum(scores) / len(scores) if scores else None
    return {'overall_grade': None if avg is None else _band(avg), 'average_score': avg,
            'graded_categories': len(scores), 'metrics': cats}
