"""Per-season coverage audit for opening game odds (and optionally props). Diagnosis only.

For each season it separates the possible causes of "no pregame line":
  games_without_tip_time      the game index lacks 'datetime' (pre-tip check impossible)
  games_without_any_row       the API returned no opening row for that game
  games_with_only_post_tip    rows exist but every row opened at/after tip
  games_with_pregame_row      usable
and reports opening lead-time quantiles (hours before tip) plus price completeness.

python -m scripts.odds_coverage_audit --season 2023 --games games_2023.jsonl --odds opening_2023.jsonl \
   --season 2024 --games ... --odds ... --out research_results/coverage/game_odds_coverage.json
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from v13.pricing import parse_utc, pregame_status, tip_times


def read(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def audit_season(games, rows, kind="game"):
    regular = [g for g in games if g.get("postseason") is False or g.get("postseason") is None]
    tips = tip_times(regular)
    ids = {str(g["id"]) for g in regular}
    by_game = defaultdict(list)
    foreign = 0
    for r in rows:
        gid = str(r.get("game_id"))
        if gid in ids:
            by_game[gid].append(r)
        else:
            foreign += 1
    status_games = Counter()
    leads, row_status = [], Counter()
    priced = Counter()
    for gid in ids:
        rs = by_game.get(gid, [])
        if gid not in tips:
            status_games["games_without_tip_time"] += 1
        if not rs:
            status_games["games_without_any_row"] += 1
            continue
        st = [pregame_status(r.get("opened_at"), tips.get(gid)) for r in rs]
        row_status.update(st)
        if "pregame" in st:
            status_games["games_with_pregame_row"] += 1
        elif all(s == "post_tip" for s in st):
            status_games["games_with_only_post_tip"] += 1
        else:
            status_games["games_with_only_unverifiable"] += 1
        for r, s in zip(rs, st):
            if s == "pregame":
                leads.append((tips[gid] - parse_utc(r["opened_at"])).total_seconds() / 3600)
            if kind == "game":
                priced["rows_with_spread_price"] += r.get("spread_home_odds") is not None and r.get("spread_away_odds") is not None
                priced["rows_with_total_price"] += r.get("total_over_odds") is not None and r.get("total_under_odds") is not None
            else:
                m = r.get("market") or {}
                priced["rows_with_two_sided_price"] += m.get("over_odds") is not None and m.get("under_odds") is not None
    q = np.percentile(leads, [5, 25, 50, 75, 95]).round(1).tolist() if leads else None
    return {"regular_games": len(ids), "rows": len(rows), "rows_for_other_games": foreign,
            **dict(status_games), "row_status": dict(row_status), **dict(priced),
            "pregame_lead_hours_p5_p25_p50_p75_p95": q,
            "vendors": dict(Counter(str(r.get("vendor")) for rs in by_game.values() for r in rs))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, action="append", required=True)
    p.add_argument("--games", action="append", required=True)
    p.add_argument("--odds", action="append", required=True)
    p.add_argument("--kind", choices=("game", "prop"), default="game")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if not len(a.season) == len(a.games) == len(a.odds):
        raise SystemExit("Give --season/--games/--odds once per season")
    report = {str(s): audit_season(read(g), read(o), a.kind) for s, g, o in zip(a.season, a.games, a.odds)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
