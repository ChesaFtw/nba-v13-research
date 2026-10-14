"""Real sportsbook price handling and point-in-time line guards.

BALLDONTLIE v2 opening records carry actual American prices:
  player props: row["market"] = {"type": "over_under", "over_odds": int, "under_odds": int}
  game odds:    spread_home_odds, spread_away_odds, total_over_odds, total_under_odds
Historical backtests must grade at these prices. -110 is only a labelled fallback.
"""
import math
from datetime import datetime, timezone

ASSUMED_110 = -110


def _american(value):
    """Return a valid American price as int, or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(odds) or -100 < odds < 100:
        return None
    return int(round(odds))


def profit_per_unit(odds):
    """Profit on a 1-unit winning stake at American odds."""
    odds = _american(odds)
    if odds is None:
        raise ValueError("Invalid American odds")
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def implied_probability(odds):
    """Break-even win probability at American odds (includes vig)."""
    return 1.0 / (1.0 + profit_per_unit(odds))


def no_vig_pair(odds_a, odds_b):
    """Multiplicatively de-vigged probabilities for a two-way market."""
    a, b = implied_probability(odds_a), implied_probability(odds_b)
    total = a + b
    return a / total, b / total


def prop_prices(row):
    """(over_odds, under_odds) from a BALLDONTLIE prop record, or (None, None)."""
    market = row.get("market") or {}
    if market.get("type") != "over_under":
        return None, None
    return _american(market.get("over_odds")), _american(market.get("under_odds"))


GAME_PRICE_FIELDS = {
    ("spread", "home"): "spread_home_odds",
    ("spread", "away"): "spread_away_odds",
    ("total", "over"): "total_over_odds",
    ("total", "under"): "total_under_odds",
}


def game_price(row, market, side):
    return _american(row.get(GAME_PRICE_FIELDS[(market, side)]))


def game_price_pair(row, market):
    sides = ("home", "away") if market == "spread" else ("over", "under")
    return tuple(game_price(row, market, s) for s in sides)


def parse_utc(value):
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        return None
    return stamp.astimezone(timezone.utc)


def tip_times(games):
    """game_id -> UTC tip time from BALLDONTLIE game 'datetime' field."""
    out = {}
    for g in games:
        stamp = parse_utc(g.get("datetime"))
        if stamp is not None and g.get("id") is not None:
            out[str(g["id"])] = stamp
    return out


def pregame_status(opened_at, tip):
    """'pregame', 'post_tip', or 'unverifiable' for a line timestamp."""
    opened = parse_utc(opened_at)
    if opened is None or tip is None:
        return "unverifiable"
    return "pregame" if opened < tip else "post_tip"


def wilson(wins, n, z=1.96):
    """95% Wilson interval for a win rate."""
    if n == 0:
        return None, None
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(centre - half, 4), round(centre + half, 4)
