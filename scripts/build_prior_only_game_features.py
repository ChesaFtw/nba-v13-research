"""Season-wide leakage-safe rolling NBA features from finished games.

For every date, write game inputs BEFORE updating history with that date's
results. Player eligibility candidates come from PREVIOUS known team rosters,
never actual future-game participation, which would be hindsight leakage.
"""
from collections import defaultdict
from datetime import date
from pathlib import Path
import json

ROOT = Path("research_data/live_nba")
OUT = Path("research_data/derived_prior_game/pregame_features.json")
TEAM_KEYS = (
    "score", "points_allowed", "estimated_possessions", "effective_fg_pct",
    "free_throw_rate", "turnover_rate_est", "off_rating_est", "def_rating_est",
    "three_point_attempt_rate", "offensive_rebounds",
    "points_in_paint", "fastbreak_points", "fouls", "free_throw_attempts",
)
PLAYER_KEYS = (
    "minutes", "points", "assists", "reboundsTotal", "reboundsOffensive",
    "fieldGoalsAttempted", "freeThrowsAttempted", "threePointersMade",
    "threePointersAttempted", "turnovers", "plusMinusPoints",
)


def _numeric(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def avg(rows, name):
    nums = [_numeric(r.get(name)) for r in rows]
    nums = [x for x in nums if x is not None]
    return round(sum(nums) / len(nums), 4) if nums else None


def summarize(history, keys, target_day: str):
    valid = [r for r in history if r["game_date"] < target_day]
    if len(valid) != len(history):
        raise ValueError("Lookahead contamination detected")
    rest = (date.fromisoformat(target_day) -
            date.fromisoformat(valid[-1]["game_date"])).days - 1 if valid else None
    return {"prior_game_count": len(valid), "rest_days": rest,
            "prior5": {k: avg(valid[-5:], k) for k in keys},
            "prior10": {k: avg(valid[-10:], k) for k in keys}}


def _team_game_features(team, opp):
    stats = team.get("statistics") or {}
    raw = team.get("four_factor_raw") or {}
    field = _numeric(raw.get("field_goals_attempted")) or 0
    three_attempts = _numeric(raw.get("three_pointers_attempted")) or 0
    rate = round(three_attempts / field, 4) if field else None
    poss = (team["estimated_possessions"] + opp["estimated_possessions"]) / 2
    if poss <= 0:
        raise ValueError("Invalid game possession estimate")
    tri = team["tricode"]
    return {
        "game_date": team["game_date"] if "game_date" in team else "",
        "score": team["score"], "points_allowed": opp["score"],
        "estimated_possessions": round(poss, 3),
        "effective_fg_pct": team.get("effective_fg_pct"),
        "free_throw_rate": team.get("free_throw_rate"),
        "turnover_rate_est": team.get("turnover_rate_est"),
        "off_rating_est": round(team["score"] / poss * 100, 3),
        "def_rating_est": round(opp["score"] / poss * 100, 3),
        "three_point_attempt_rate": rate,
        "offensive_rebounds": _numeric(raw.get("offensive_rebounds")),
        "points_in_paint": _numeric(stats.get("pointsInPaint")),
        "fastbreak_points": _numeric(stats.get("fastBreakPoints")),
        "fouls": _numeric(stats.get("fouls")),
        "free_throw_attempts": _numeric(raw.get("free_throws_attempted")),
        "prior_roster_player_ids": [p["player_id"] for p in team.get("players") or [] if p.get("played")],
        "prior_starter_player_ids": [p["player_id"] for p in team.get("players") or [] if p.get("starter")],
    }


def build_prior_only(games):
    unique = {}
    for g in games:
        if g.get("game_status") != "final":
            raise ValueError("Unfinished games may not create model features")
        if g["game_id"] in unique:
            raise ValueError("Duplicate game records; reconcile upstream")
        unique[g["game_id"]] = g
    by_date = defaultdict(list)
    for g in unique.values():
        by_date[g["game_date"]].append(g)
    team_hist, player_hist = defaultdict(list), defaultdict(list)
    records = []
    for day in sorted(by_date):
        todays = sorted(by_date[day], key=lambda g: g["game_id"])
        for g in todays:
            out = {
                "game_id": g["game_id"], "game_date": day,
                "feature_asof_exclusive": day,
                "away_tricode": g["away"]["tricode"],
                "home_tricode": g["home"]["tricode"],
                "teams": {}, "previous_roster_players": {},
            }
            for team in (g["away"], g["home"]):
                tri = team["tricode"]
                hist = team_hist[tri]
                out["teams"][tri] = summarize(hist, TEAM_KEYS, day)
                prev = hist[-1] if hist else {}
                prior_roster = prev.get("prior_roster_player_ids") or []
                out["teams"][tri]["previous_roster_ids"] = prior_roster
                out["teams"][tri]["previous_starter_ids"] = prev.get("prior_starter_player_ids") or []
                for player_id in prior_roster:
                    out["previous_roster_players"][player_id] = summarize(
                        player_hist[player_id], PLAYER_KEYS, day)
            # Outcome labels are deliberately separated and NEVER valid predictors.
            out["postgame_label_for_evaluation_only"] = {
                "actual_total": g["total"], "actual_home_margin": g["home_margin"]}
            records.append(out)
        for g in todays:
            away, home = g["away"], g["home"]
            for team, opp in ((away, home), (home, away)):
                t = _team_game_features(team, opp)
                t["game_date"] = day
                team_hist[team["tricode"]].append(t)
                for p in team["players"]:
                    if not p.get("played") or not p.get("player_id"):
                        continue
                    d = {"game_date": day}
                    d.update({key: p.get(key) for key in PLAYER_KEYS})
                    player_hist[p["player_id"]].append(d)
    return records


def main():
    games = [json.loads(path.read_text(encoding="utf-8"))
             for path in sorted(ROOT.glob("*/[0-9]*.json"))]
    if not games:
        raise ValueError("No finished archived games; pregame features unavailable")
    output = build_prior_only(games)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Built {len(output)} pregame feature records, same-date game outcomes excluded")


if __name__ == "__main__":
    main()
