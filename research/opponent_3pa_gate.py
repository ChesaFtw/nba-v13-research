"""Research-only, appearance-conditional lagged opponent 3PA allowance guard.

No model activation, sportsbook prices, staking, or wager execution.
"""
from datetime import date
from math import isfinite


def opponent_3pa_factor(history, league_3pa_per48, target_date, strength=1.0,
                        prior_games=20.0, min_games=8):
    """Return bounded opponent allowance factor from strictly earlier team games."""
    try:
        target = date.fromisoformat(target_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid target date") from exc
    if (isinstance(league_3pa_per48, bool) or not isinstance(league_3pa_per48, (int, float))
            or not isfinite(league_3pa_per48) or not 10 <= league_3pa_per48 <= 65):
        raise ValueError("invalid prior league three-point rate")
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not isfinite(strength) or not 0 <= strength <= 2):
        raise ValueError("invalid strength")
    if (isinstance(prior_games, bool) or not isinstance(prior_games, (int, float))
            or not isfinite(prior_games) or prior_games <= 0):
        raise ValueError("invalid shrinkage")
    if type(min_games) is not int or min_games < 1:
        raise ValueError("invalid minimum sample")
    previous = None
    rates = []
    for row in history:
        try:
            day = date.fromisoformat(row["date"])
            rate = row["opponent_3pa_per48"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid opponent history") from exc
        if day >= target or (previous is not None and day <= previous):
            raise ValueError("opponent history not strictly lagged and ordered")
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not isfinite(rate) or not 0 <= rate <= 80):
            raise ValueError("invalid opponent three-point attempts")
        rates.append(float(rate))
        previous = day
    if len(rates) < min_games:
        return 1.0
    n = len(rates)
    raw = sum(rates) / n / league_3pa_per48 - 1.0
    factor = 1.0 + strength * (n / (n + prior_games)) * raw
    return min(1.15, max(0.85, factor))
