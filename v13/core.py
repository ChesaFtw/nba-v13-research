"""Transparent screening primitives. No claimed predictive edge."""
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class Candidate:
    game_id: str
    market: str
    selection: str
    predicted_probability: float
    decimal_odds: float
    as_of_utc: str

def implied_probability(decimal_odds: float) -> float:
    if not isfinite(decimal_odds) or decimal_odds <= 1:
        raise ValueError("Decimal odds must exceed 1")
    return 1.0 / decimal_odds

def expected_return(probability: float, decimal_odds: float) -> float:
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")
    return probability * (decimal_odds - 1) - (1 - probability)

def american_to_decimal(odds: int) -> float:
    if odds == 0 or -100 < odds < 100:
        raise ValueError("Valid American odds are <= -100 or >= +100")
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))

def evaluate(candidate: Candidate) -> dict:
    return {
        "game_id": candidate.game_id,
        "market": candidate.market,
        "selection": candidate.selection,
        "as_of_utc": candidate.as_of_utc,
        "implied_probability": implied_probability(candidate.decimal_odds),
        "expected_return": expected_return(candidate.predicted_probability, candidate.decimal_odds),
        "status": "RESEARCH_ONLY_UNVALIDATED",
    }
