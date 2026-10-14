"""Leakage-aware historical player prop features and next-game baseline projections."""
from collections import defaultdict
from datetime import datetime

MARKETS = {
    "points": ("pts",),
    "rebounds": ("reb",),
    "assists": ("ast",),
    "threes": ("fg3m",),
    "pra": ("pts", "reb", "ast"),
}

def _date(row):
    game = row.get("game") or {}
    return str(game.get("date") or "")[:10]

def _value(row, fields):
    return sum(float(row.get(field) or 0) for field in fields)

def build_prop_features(rows, min_history=5):
    """Yield one pregame feature record per player-game, ordered chronologically.

    All rolling features exclude the target game. This is a research baseline,
    not a betting recommendation or a calibrated probability.
    """
    history = defaultdict(list)
    ordered = sorted(rows, key=lambda r: (_date(r), (r.get("game") or {}).get("id") or 0, r.get("id") or 0))
    for row in ordered:
        player = row.get("player") or {}
        pid = player.get("id")
        game = row.get("game") or {}
        if pid is None or not _date(row):
            continue
        prior = [r for r in history[pid] if _date(r) < _date(row)]
        if len(prior) >= min_history:
            result = {"player_id": pid, "game_id": game.get("id"), "game_date": _date(row), "history_games": len(prior)}
            for market, fields in MARKETS.items():
                for window in (5, 10, 20):
                    sample = prior[-window:]
                    result[f"{market}_avg_{window}"] = round(sum(_value(r, fields) for r in sample) / len(sample), 3)
                result[f"{market}_actual"] = _value(row, fields)
            yield result
        history[pid].append(row)
