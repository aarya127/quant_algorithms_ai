from dataclasses import dataclass
from typing import Dict, List

from .execution_simulator import OptionFillSimulator
from .portfolio_constructor import HedgePolicy, PortfolioBook, PortfolioConstructor
from .position_sizer import GreekAwarePositionSizer
from .schemas import (
    ExecutionAssumptions,
    OptionPositionCandidate,
    PortfolioConstraints,
    SizingBasis,
    TargetPosition,
)


@dataclass
class PortfolioBuildResult:
    candidates: List[OptionPositionCandidate]
    targets: List[TargetPosition]
    book: PortfolioBook
    fill_results: list


def parse_action_summary(action_summary: str):
    s = action_summary.upper()
    side = "buy" if "BUY" in s else "sell"
    option_type = "call" if "CALL" in s else "put"
    return side, option_type


def build_candidates_from_intents(
    intents: list,
    signals_by_intent_id: Dict[str, object],
    market_by_strike: Dict[float, Dict],
    spot: float,
) -> List[OptionPositionCandidate]:
    """
    Build candidate positions from intents plus available signal and market metadata.

    Note: Greek values are research-grade proxies when direct model greeks are not supplied.
    """
    out: List[OptionPositionCandidate] = []

    for intent in intents:
        signal = signals_by_intent_id.get(intent.intent_id)
        md = market_by_strike.get(intent.strike, {})
        bid = float(md.get("bid", 0.01))
        ask = float(md.get("ask", max(bid, 0.01)))
        mid = float(md.get("mid", (bid + ask) / 2.0))
        vol = int(md.get("volume", 0))
        oi = int(md.get("open_interest", 0))

        side, option_type = parse_action_summary(intent.action_summary)

        # Simple proxy greeks for sizing if no direct greeks are available.
        t = max(float(intent.maturity_days) / 365.0, 1.0 / 365.0)
        moneyness = intent.strike / max(spot, 1e-9)
        vega = max(5.0, 20.0 * (t ** 0.5) * (1.0 - min(abs(moneyness - 1.0), 0.3)))
        gamma = max(0.05, 2.0 * (1.0 / max(spot, 1.0)) * (1.0 / max(t ** 0.5, 0.08)))
        theta = max(1.0, 0.10 * max(mid * 100.0, 1.0) / max(intent.holding_horizon_days, 1))
        delta = 0.50 if option_type == "call" else -0.50
        if side == "sell":
            delta *= -1.0

        spread = max(ask - bid, 0.0)
        stress_loss = max(mid * 100.0 + spread * 100.0, 1.0)

        out.append(
            OptionPositionCandidate(
                intent_id=intent.intent_id,
                ticker=intent.ticker,
                option_type=option_type,
                side=side,
                strike=float(intent.strike),
                maturity_days=float(intent.maturity_days),
                confidence=float(intent.confidence),
                expected_net_edge_bps=float(intent.expected_net_edge_bps),
                bid=bid,
                ask=ask,
                mid=mid,
                volume=vol,
                open_interest=oi,
                delta=delta,
                gamma=gamma,
                vega=vega,
                theta=theta,
                stress_loss_per_contract=stress_loss,
                quote_timestamp=None,
            )
        )

    return out


def build_portfolio_from_candidates(
    candidates: List[OptionPositionCandidate],
    constraints: PortfolioConstraints,
    sizing_basis: SizingBasis,
    hedge_policy: HedgePolicy,
    execution: ExecutionAssumptions,
    allow_net_vega_accumulation: bool = True,
) -> PortfolioBuildResult:
    sizer = GreekAwarePositionSizer(constraints=constraints, basis=sizing_basis)
    targets = sizer.size(candidates)

    constructor = PortfolioConstructor(
        hedge_policy=hedge_policy,
        delta_threshold=25.0,
        allow_net_vega_accumulation=allow_net_vega_accumulation,
    )
    book = constructor.construct(targets)

    filler = OptionFillSimulator(execution)
    candidate_lookup = {c.intent_id: c for c in candidates}
    fills = filler.simulate(book.positions, candidate_lookup)

    return PortfolioBuildResult(
        candidates=candidates,
        targets=targets,
        book=book,
        fill_results=fills,
    )
