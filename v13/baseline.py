"""Leak-free chronological Elo baseline for NBA game winners.

This is a benchmark, not a betting strategy or calibrated confidence model.
"""
import math
from collections import defaultdict

HOME_ADVANTAGE = 65.0
K_FACTOR = 20.0
SEASON_REGRESSION = 0.75


def _team_id(value):
    if isinstance(value, dict):
        return value.get("id")
    return value


def _finished(game):
    home = game.get("home_team_score")
    away = game.get("visitor_team_score")
    return (
        isinstance(home, (int, float)) and isinstance(away, (int, float))
        and home != away and home >= 0 and away >= 0
        and game.get("status") == "Final"
    )


def evaluate_seasons(seasons):
    """Evaluate each game's pre-tipoff prediction, then update ratings.

    seasons: iterable of (season_year, games), ordered chronologically.
    """
    ratings = defaultdict(lambda: 1500.0)
    records = []
    last_season = None
    for season, games in sorted(seasons, key=lambda x: x[0]):
        if last_season is not None:
            ratings = defaultdict(lambda: 1500.0, {
                team: 1500.0 + (rating - 1500.0) * SEASON_REGRESSION
                for team, rating in ratings.items()
            })
        last_season = season
        for game in sorted(games, key=lambda g: (str(g.get("date", "")), str(g.get("id", "")))):
            if not _finished(game):
                continue
            home = _team_id(game.get("home_team"))
            away = _team_id(game.get("visitor_team"))
            if home is None or away is None or home == away:
                continue
            h = ratings[home]
            a = ratings[away]
            p_home = 1.0 / (1.0 + 10.0 ** (-(h + HOME_ADVANTAGE - a) / 400.0))
            outcome = int(game["home_team_score"] > game["visitor_team_score"])
            records.append({
                "season": season, "game_id": game.get("id"),
                "date": game.get("date"), "home_team_id": home,
                "visitor_team_id": away, "p_home": round(p_home, 6),
                "actual_home_win": outcome,
                "predicted_home_win": int(p_home >= 0.5),
            })
            delta = K_FACTOR * (outcome - p_home)
            ratings[home] = h + delta
            ratings[away] = a - delta
    return records


def summarize(records):
    if not records:
        return {"games": 0, "accuracy": None, "brier": None}
    n = len(records)
    return {
        "games": n,
        "accuracy": round(sum(r["predicted_home_win"] == r["actual_home_win"] for r in records) / n, 4),
        "brier": round(sum((r["p_home"] - r["actual_home_win"]) ** 2 for r in records) / n, 5),
    }
