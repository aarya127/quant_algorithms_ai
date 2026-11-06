"""
Cash-flow and capital lines from SEC filings for the Statistics tab
(Finnhub "financials as reported" — free tier, US filers only).

A 10-Q reports year-to-date figures, so the trailing twelve months are
    this year's YTD + the last 10-K - last year's YTD to the same quarter.
When the year-ago pieces are missing, the latest 10-K is used as is (basis
'annual'). Balance-sheet lines come from the latest filing that has them
(some 10-Qs arrive with an empty balance sheet).
"""
import datetime

# first concept present wins (companies tag the same line differently)
FLOWS = {
    'revenue': ('ic', ['us-gaap_Revenues', 'us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax',
                       'us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax',
                       'us-gaap_SalesRevenueNet', 'us-gaap_SalesRevenueGoodsNet']),
    'operating_income': ('ic', ['us-gaap_OperatingIncomeLoss']),
    'operating_cash_flow': ('cf', ['us-gaap_NetCashProvidedByUsedInOperatingActivities',
                                   'us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations']),
    'capex': ('cf', ['us-gaap_PaymentsToAcquirePropertyPlantAndEquipment',
                     'us-gaap_PaymentsToAcquireProductiveAssets',
                     'us-gaap_PaymentsForCapitalImprovements']),
    'stock_comp': ('cf', ['us-gaap_ShareBasedCompensation',
                          'us-gaap_AllocatedShareBasedCompensationExpense']),
    'depreciation_amortization': ('cf', ['us-gaap_DepreciationDepletionAndAmortization',
                                         'us-gaap_DepreciationAmortizationAndAccretionNet',
                                         'us-gaap_DepreciationAndAmortization']),
    'buybacks': ('cf', ['us-gaap_PaymentsForRepurchaseOfCommonStock',
                        'us-gaap_PaymentsForRepurchaseOfEquity']),
    'dividends_paid': ('cf', ['us-gaap_PaymentsOfDividends', 'us-gaap_PaymentsOfDividendsCommonStock']),
}
# D&A tagged only as separate lines (e.g. AMD) — summed when no total is tagged
DA_PARTS = ['us-gaap_Depreciation', 'us-gaap_OtherDepreciationAndAmortization',
            'us-gaap_AmortizationOfIntangibleAssets']
# outflows reported as positive payments; abs() guards a filer that signs them
OUTFLOWS = ('capex', 'buybacks', 'dividends_paid')
INSTANTS = {
    'total_assets': ('bs', ['us-gaap_Assets']),
    'current_liabilities': ('bs', ['us-gaap_LiabilitiesCurrent']),
}


def _number(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float('inf') else None


def _lines(report, section):
    return {x.get('concept'): _number(x.get('value')) for x in report.get(section) or []}


def _first(lines, concepts):
    return next((lines[c] for c in concepts if lines.get(c) is not None), None)


def parse_filing(rep):
    """One Finnhub as-reported filing → (fy, fq, end date, values); fq 0 = 10-K."""
    report = rep.get('report') or {}
    sections = {s: _lines(report, s) for s in ('ic', 'cf', 'bs')}
    vals = {name: _first(sections[sec], concepts)
            for name, (sec, concepts) in {**FLOWS, **INSTANTS}.items()}
    if vals['depreciation_amortization'] is None:
        parts = [sections['cf'][c] for c in DA_PARTS if sections['cf'].get(c) is not None]
        vals['depreciation_amortization'] = sum(parts) if parts else None
    for name in OUTFLOWS:
        if vals[name] is not None:
            vals[name] = abs(vals[name])
    end = datetime.date.fromisoformat(str(rep['endDate'])[:10])
    return int(rep['year']), int(rep.get('quarter') or 0), end, vals


def latest_ttm(quarterly, annual):
    """Trailing-twelve-month flows and latest balance-sheet lines, or {} when
    there are no usable filings. Keys: FLOWS, INSTANTS, free_cash_flow,
    basis ('ttm' | 'annual'), period_end (YYYY-MM-DD)."""
    filings = {}
    for rep, is_annual in [(r, False) for r in quarterly or []] + [(r, True) for r in annual or []]:
        try:
            fy, fq, end, vals = parse_filing(rep)
        except (KeyError, TypeError, ValueError):
            continue
        if is_annual:
            fq = 0
        elif fq not in (1, 2, 3):
            continue
        filings[(fy, fq)] = (end, vals)
    if not filings:
        return {}

    (fy, fq), (end, latest) = max(filings.items(), key=lambda kv: kv[1][0])
    prev_annual, prev_ytd = filings.get((fy - 1, 0)), filings.get((fy - 1, fq))
    # the year-ago quarter must really be a year earlier (fiscal-year labels can drift)
    year_ago_ok = prev_ytd is not None and abs((end - prev_ytd[0]).days - 365) <= 45
    if fq == 0:
        basis, out = 'annual', {k: latest[k] for k in FLOWS}
    elif prev_annual and year_ago_ok:
        basis = 'ttm'
        out = {k: (latest[k] + prev_annual[1][k] - prev_ytd[1][k]
                   if None not in (latest[k], prev_annual[1][k], prev_ytd[1][k]) else None)
               for k in FLOWS}
    else:
        annuals = [(e, v) for (_, q), (e, v) in filings.items() if q == 0]
        if not annuals:
            return {}
        end, latest_annual = max(annuals, key=lambda ev: ev[0])
        basis, out = 'annual', {k: latest_annual[k] for k in FLOWS}

    for name in INSTANTS:   # newest filing that tagged it
        out[name] = next((v[name] for e, v in sorted(filings.values(), key=lambda ev: ev[0], reverse=True)
                          if v[name] is not None), None)
    ocf, capex = out['operating_cash_flow'], out['capex']
    out['free_cash_flow'] = ocf - capex if ocf is not None and capex is not None else None
    out['basis'] = basis
    out['period_end'] = end.isoformat()
    return out


def get_cash_flow_summary(symbol: str) -> dict:
    """latest_ttm() for a symbol from Finnhub (2 calls; cache the result)."""
    from data.finnhub import get_financials_reported
    return latest_ttm(get_financials_reported(symbol, 'quarterly'),
                      get_financials_reported(symbol, 'annual'))
