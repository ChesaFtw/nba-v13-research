"""Chronological pregame feature store, strictly prior dates only.

These features are computed from finalized NBA games. They are NOT predictions,
and same-game player minutes or starters must never enter pregame features.
"""
from collections import defaultdict
from datetime import date
from pathlib import Path
import json

ROOT = Path("research_data/live_nba")
OUT = Path("research_data/derived_prior_game/pregame_features.json")
TEAM_STATS = (
    "score", "estimated_possessions", "effective_fg_pct",
    "free_throw_rate", "turnover_rate_est",
)
PLAYER_STATS = (
    "points", "assists", "reboundsTotal", "reboundsOffensive",
    "threePointersMade", "threePointersAttempted", "fieldGoalsAttempted",
    "freeThrowsAttempted", "turnovers", "plusMinusPoints",
)


def avg(rows, name):
    nums = [float(r[name]) for r in rows if isinstance(r.get(name), (float, int))]
    return round(sum(nums) / len(nums), 4) if nums else None


def prior_summary(history, keys, today: str):
    valid = [r for r in history if r["game_date"] < today]
    if len(valid) != len(history):
        raise ValueError("Lookahead contamination in history")
    last = valid[-5:]
    rest = (date.fromisoformat(today) - date.fromisoformat(valid[-1]["game_date"])).days - 1 if valid else None
    return {"prior_game_count": len(valid), "days_off_since_previous_game": rest,
            "prior5": {k: avg(last, k) for k in keys},
            "prior10": {k: avg(valid[-10:], k) for k in keys}}


def build_prior_only(games):
    unique = {}
    for g in games:
        if g.get("game_status") != "final":
            raise ValueError("Unfinished game in feature set")
        gid = str(g["game_id"])
        if gid in unique and unique[gid] != g:
            raise ValueError("Conflicting game IDs")
        unique[gid] = g
    by_date = defaultdict(list)
    for g in unique.values():
        by_date[g["game_date"]].append(g)
    team_hist, player_hist = defaultdict(list), defaultdict(list)
    records = []
    # Crucial: write features for ALL games on a date before adding ANY results for that date
    for day in sorted(by_date):
        todays = sorted(by_date[day], key=lambda g: g["game_id"])
        for g in todays:
            result = {"game_id": g["game_id"], "game_date": day, "feature_asof_exclusive": day,
                      "away_tricode": g["away"]["tricode"], "home_tricode": g["home"]["tricode"],
                      "teams": {}, "players_prior_only": {}}
            for team in (g["away"], g["home"]):
                tri = team["tricode"]
                result["teams"][tri] = prior_summary(team_hist[tri], TEAM_STATS, day)
                # These are labels for later evaluation, not pregame features.
            for team in (g["away"], g["home"]):
                for p in team["players"]:
                    pid = p["player_id"]
                    result["players_prior_only"][pid] = prior_summary(player_hist[pid], PLAYER_STATS, day)
            result["postgame_label_for_evaluation_only"] = {
                "actual_total": g["total"], "actual_home_margin": g["home_margin"]}
            records.append(result)
        # update only after features for every same-date game were produced
        for g in todays:
            for team in (g["away"], g["home"]):
                tri = team["tricode"]
                team_hist[tri].append({"game_date": day, **{k: team.get(k) for k in TEAM_STATS}})
                for p in team["players"]:
                    if not p.get("played"):
                        continue
                    pid = p["player_id"]
                    player_hist[pid].append({"game_date": day, **{k: p.get(k) for k in PLAYER_STATS}})
    return records


def main():
    games = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ROOT.glob("*/[0-9]*.json"))]
    if not games:
        raise ValueError("No ingested final games yet; cannot create research features")
    result = build_prior_only(games)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {len(result)} leakage-safe game feature records to {OUT}")


if __name__ == "__main__":
    main()
