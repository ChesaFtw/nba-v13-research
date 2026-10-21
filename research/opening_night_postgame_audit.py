"""Opening-night wager audit and pre-bet screening. No model fitting or hindsight tuning."""
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Wager:
    label: str
    stake: float
    american_odds: int
    outcome: str  # win, loss, pending
    market: str
    line: float
    actual: Optional[float] = None
    model_probability: Optional[float] = None
    validated: bool = False

def break_even_probability(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return (-odds / (-odds + 100)) if odds < 0 else (100 / (odds + 100))

def realized_profit(wager: Wager) -> float:
    if wager.outcome == "loss":
        return -wager.stake
    if wager.outcome == "win":
        return wager.stake * (100 / -wager.american_odds if wager.american_odds < 0 else wager.american_odds / 100)
    if wager.outcome == "pending":
        return 0.0
    raise ValueError("Unknown outcome")

def bet_gate(wager: Wager, min_edge: float = 0.03) -> tuple[bool, str]:
    """Never promote anecdotal picks or unvalidated probabilities to official bets."""
    if wager.model_probability is None or not wager.validated:
        return False, "NO_BET: missing out-of-sample validated probability"
    if not 0 <= wager.model_probability <= 1:
        return False, "NO_BET: invalid probability"
    edge = wager.model_probability - break_even_probability(wager.american_odds)
    if edge < min_edge:
        return False, "NO_BET: insufficient probability edge"
    return True, "ELIGIBLE: requires current quote, lineup, and availability verification"

OPENING_NIGHT = (
    Wager("Tatum rebounds over", 100, -121, "loss", "rebounds_over", 8.5, 4),
    Wager("OKC-SAS total under", 100, -110, "win", "total_under", 231.5, 220),
)

def summary() -> dict:
    return {
        "bets": len(OPENING_NIGHT),
        "wins": sum(w.outcome == "win" for w in OPENING_NIGHT),
        "losses": sum(w.outcome == "loss" for w in OPENING_NIGHT),
        "profit": round(sum(realized_profit(w) for w in OPENING_NIGHT), 2),
        "wagers": [
            {"label": w.label, "actual": w.actual, "line": w.line,
             "profit": round(realized_profit(w), 2),
             "break_even_probability": round(break_even_probability(w.american_odds), 5),
             "gate": bet_gate(w)[1]}
            for w in OPENING_NIGHT
        ],
    }

if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
