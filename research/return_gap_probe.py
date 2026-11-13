"""Exploratory appearance-conditional team-game-gap minutes probe.

Use completed regular-season games and played-player box scores. Cannot
establish prospective player availability or injury cause.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import mean

def _minutes(value):
    if not isinstance(value, str):
        return float(value or 0)
    if ":" in value:
        a, b = value.split(":", 1)
        return float(a) + float(b) / 60
    return float(value)

def build_rows(games, players, min_prior=10):
    if min_prior < 5:
        raise ValueError("min_prior must be at least five")
    team_games = defaultdict(list)
    for game in games:
        if game.get("postseason") or game.get("status") != "Final":
            continue
        for side in ("home_team", "visitor_team"):
            team_games[game[side]["id"]].append((game["date"], game["id"]))
    team_index = {
        team: {gid: i for i, (_, gid) in enumerate(sorted(games))}
        for team, games in team_games.items()
    }
    history = defaultdict(list)
    output = []
    ordered = sorted(players, key=lambda r: (r["game"]["date"], r["game"]["id"], r["player"]["id"]))
    for r in ordered:
        pid, team, gid = r["player"]["id"], r["team"]["id"], r["game"]["id"]
        if gid not in team_index.get(team, {}):
            continue
        minutes = _minutes(r.get("min"))
        if minutes <= 0:
            continue
        idx = team_index[team][gid]
        key = (pid, team)
        prev = history[key]
        if prev and idx <= prev[-1][0]:
            raise ValueError("duplicate or unordered player/team game")
        if len(prev) >= min_prior:
            gap = idx - prev[-1][0] - 1
            last5 = mean(m for _, m in prev[-5:])
            last10 = mean(m for _, m in prev[-10:])
            output.append({
                "date": r["game"]["date"], "game_id": gid, "player_id": pid,
                "gap_team_games": gap, "minutes": minutes,
                "last5": last5, "last10": last10,
                "signed_error_last10": minutes - last10,
                "abs_error_last5": abs(minutes - last5),
                "abs_error_last10": abs(minutes - last10),
            })
        prev.append((idx, minutes))
    return output

def cluster_ci(rows, field="signed_error_last10", iterations=1000, seed=42):
    games = defaultdict(list)
    for row in rows:
        games[row["game_id"]].append(row[field])
    if len(games) < 15:
        return None
    clusters = [(sum(vals), len(vals)) for vals in games.values()]
    rng = Random(seed)
    estimates = []
    for _ in range(iterations):
        draw = rng.choices(clusters, k=len(clusters))
        estimates.append(sum(s for s, _ in draw) / sum(n for _, n in draw))
    estimates.sort()
    return [round(estimates[int((iterations - 1) * p)], 3) for p in (.025, .975)]

def summarize(rows):
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows), "games": len({r["game_id"] for r in rows}),
        "avg_missed_team_games": round(mean(r["gap_team_games"] for r in rows), 2),
        "mean_minutes_minus_last10": round(mean(r["signed_error_last10"] for r in rows), 3),
        "last5_mae": round(mean(r["abs_error_last5"] for r in rows), 3),
        "last10_mae": round(mean(r["abs_error_last10"] for r in rows), 3),
        "residual_game_cluster_ci95": cluster_ci(rows),
    }

def report(games, players):
    rows = build_rows(games, players)
    windows = [("2026-01-15", "2026-02-15"), ("2026-02-15", "2026-03-12"),
               ("2026-03-12", "2026-04-13")]
    results = {}
    for start, end in windows:
        eligible = [r for r in rows if start <= r["date"] < end]
        results[start] = {
            "eligible_appearances": len(eligible),
            "consecutive": summarize([r for r in eligible if r["gap_team_games"] == 0]),
            "missed_1_2": summarize([r for r in eligible if 1 <= r["gap_team_games"] <= 2]),
            "missed_3plus": summarize([r for r in eligible if r["gap_team_games"] >= 3]),
        }
    return {
        "season": 2025,
        "data_scope": "appearance-conditioned, played-player box scores only",
        "limitations": "No injury attribution, prospective availability, betting lines or independently untouched season",
        "windows": results,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", required=True)
    parser.add_argument("--players", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    def load(path):
        with Path(path).open() as f:
            return [json.loads(line) for line in f if line.strip()]
    result = report(load(args.games), load(args.players))
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
