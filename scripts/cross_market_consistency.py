"""Challenger experiment: are game lines and player-points prop lines mutually consistent?

Books set the spread/total and hundreds of player props partly separately. A team's
implied points (from spread + total) should match what its players' points props add up
to. Where they disagree, one of the two is stale. This asks, strictly out of sample,
which side the result follows, and whether a frozen rule built on the gap wins at actual prices.

Gap (per team-game) = implied team total - (a + b*sum_of_points_lines + c*players_with_lines),
with a,b,c fitted only on earlier months. Rule (pre-registered): when gap >= +G, bet the OVER
on that team's points props; when gap <= -G, the UNDER. G = 1 SD of training-month gaps.

python -m scripts.cross_market_consistency --games games_2025.jsonl --game-odds opening_2025.jsonl \
  --props opening_props_2025.jsonl --stats player_stats_regular_2025.jsonl --out research_results/consistency_2025.json
"""
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.backtest_opening_spreads_totals import choose
from scripts.backtest_prop_holdout import ALIASES, VENDORS
from v13.pricing import pregame_status, profit_per_unit, prop_prices, tip_times, wilson


def read(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def _tid(x):
    return str(x.get("id")) if isinstance(x, dict) else str(x)


def player_team_before(stats):
    """(player_id, date) -> team from the player's most recent EARLIER game (pregame-knowable)."""
    hist = defaultdict(list)
    for r in stats:
        pid = (r.get("player") or {}).get("id"); g = r.get("game") or {}
        if pid is not None and g.get("date") and r.get("team"):
            hist[str(pid)].append((str(g["date"])[:10], _tid(r["team"])))
    for v in hist.values():
        v.sort()
    def lookup(pid, day):
        prior = [t for d, t in hist.get(str(pid), []) if d < day]
        return prior[-1] if prior else None
    return lookup


def points_lines(props, tips):
    best = {}
    for r in props:
        if ALIASES.get(r.get("prop_type")) != "points" or (r.get("market") or {}).get("type") != "over_under":
            continue
        if pregame_status(r.get("opened_at"), tips.get(str(r.get("game_id")))) != "pregame":
            continue
        over, under = prop_prices(r)
        if over is None or under is None:
            continue
        key = (str(r["game_id"]), str(r["player_id"]))
        v = str(r.get("vendor") or "").lower()
        rank = (VENDORS.index(v) if v in VENDORS else 99, str(r["opened_at"]))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, {"line": float(r["line_value"]), "over": over, "under": under})
    return {k: v[1] for k, v in best.items()}


def build(games, game_odds, props, stats):
    tips = tip_times(games)
    lines = choose(game_odds, tips)
    plines = points_lines(props, tips)
    team_of = player_team_before(stats)
    actual = {(str((r.get("game") or {}).get("id")), str((r.get("player") or {}).get("id"))): float(r.get("pts") or 0) for r in stats}
    by_team = defaultdict(list)
    gdate = {str(g["id"]): str(g["date"])[:10] for g in games}
    for (gid, pid), pl in plines.items():
        team = team_of(pid, gdate.get(gid, "0000"))
        if team and (gid, pid) in actual:
            by_team[(gid, team)].append({**pl, "player_id": pid, "actual": actual[(gid, pid)]})
    rows = []
    for g in games:
        gid = str(g["id"]); o = lines.get(gid)
        if not o or o.get("spread_home_value") is None or o.get("total_value") is None or g.get("home_team_score") is None:
            continue
        margin, total = -float(o["spread_home_value"]), float(o["total_value"])
        for team, implied, pts in ((_tid(g["home_team"]), (total + margin) / 2, g["home_team_score"]),
                                   (_tid(g["visitor_team"]), (total - margin) / 2, g["visitor_team_score"])):
            players = by_team.get((gid, team), [])
            if len(players) < 3:
                continue
            rows.append({"game_id": gid, "date": gdate[gid], "team": team, "implied": implied, "actual_team": float(pts),
                         "sum_lines": sum(p["line"] for p in players), "n": len(players), "players": players})
    return sorted(rows, key=lambda r: r["date"])


def run(rows, min_train=200):
    out_rows, bets = [], []
    for month in sorted({r["date"][:7] for r in rows}):
        train = [r for r in rows if r["date"][:7] < month]
        test = [r for r in rows if r["date"][:7] == month]
        if len(train) < min_train:
            continue
        X = np.array([[1, r["sum_lines"], r["n"]] for r in train]); y = np.array([r["implied"] for r in train])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        G = float(np.std(y - X @ beta))
        for r in test:
            gap = r["implied"] - float(np.array([1, r["sum_lines"], r["n"]]) @ beta)
            out_rows.append({**r, "gap": gap, "G": G})
            if abs(gap) >= G:
                side = "over" if gap > 0 else "under"
                for p in r["players"]:
                    d = p["actual"] - p["line"]
                    o = "push" if abs(d) < 1e-9 else ("win" if (d > 0) == (side == "over") else "loss")
                    bets.append({"date": r["date"], "side": side, "price": p[side], "outcome": o})
    def corr(a, b):
        return round(float(np.corrcoef(a, b)[0, 1]), 4) if len(a) > 2 else None
    gaps = [r["gap"] for r in out_rows]
    c = Counter(b["outcome"] for b in bets); n = c["win"] + c["loss"]
    staked = sum(1 for b in bets if b["outcome"] != "push")
    pnl = sum(profit_per_unit(b["price"]) if b["outcome"] == "win" else -1 for b in bets if b["outcome"] != "push")
    return {"team_games_scored": len(out_rows),
            "corr_gap_with_team_points_minus_implied": corr(gaps, [r["actual_team"] - r["implied"] for r in out_rows]),
            "corr_gap_with_player_points_minus_lines": corr(gaps, [sum(p["actual"] for p in r["players"]) - r["sum_lines"] for r in out_rows]),
            "reading_guide": "Positive corr with player points minus lines = prop lines were stale (team total right); "
                             "positive corr with team points minus implied = the game line was stale.",
            "rule_bets": len(bets), "rule_record": dict(c), "rule_hit_95ci": wilson(c["win"], n),
            "rule_roi_flat_pct": round(100 * pnl / staked, 2) if staked else None,
            "rule_by_side": {s: dict(Counter(b["outcome"] for b in bets if b["side"] == s)) for s in ("over", "under")},
            "note": "Bets on one team-game are correlated; treat the CI as optimistic. Exploratory until replicated on a second season."}


def main():
    p = argparse.ArgumentParser()
    for k in ("games", "game-odds", "props", "stats", "out"):
        p.add_argument(f"--{k}", required=True)
    a = p.parse_args()
    rows = build(read(a.games), read(a.game_odds), read(a.props), read(a.stats))
    result = run(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
