"""Filter historical box scores against a verified regular-season game index.

Fail closed on incomplete, conflicting, or duplicate records. The player-stat
endpoint may contain postseason and DNP rows even when pagination completes.
"""
import argparse
import json
from pathlib import Path


def minutes(value):
    if value is None or str(value).strip() == "":
        return 0.0
    s = str(value)
    try:
        if ":" in s:
            m, sec = s.split(":", 1)
            return float(m) + float(sec) / 60
        return float(s)
    except ValueError:
        return 0.0


def filter_regular(games, stats, season):
    valid = {}
    for game in games:
        if game.get("season") != season or game.get("postseason") is not False:
            continue
        gid = game.get("id")
        date = str(game.get("date") or "")[:10]
        if gid is None or len(date) != 10:
            raise ValueError("Invalid regular-season game index entry")
        gid = str(gid)
        if gid in valid:
            raise ValueError("Duplicate regular-season game ID")
        valid[gid] = date
    if len(valid) != 1230:
        raise ValueError(f"Expected 1230 regular games; found {len(valid)}")

    out, seen, covered = [], set(), set()
    for row in stats:
        game = row.get("game") or {}
        gid = str(game.get("id"))
        if gid not in valid or minutes(row.get("min")) <= 0:
            continue
        if str(game.get("date") or "")[:10] != valid[gid]:
            raise ValueError("Player-game date mismatch")
        if game.get("season") is not None and game["season"] != season:
            raise ValueError("Player-game season mismatch")
        pid = (row.get("player") or {}).get("id")
        if pid is None:
            raise ValueError("Missing player ID in played regular-season row")
        key = (gid, pid)
        if key in seen:
            raise ValueError("Duplicate player-game")
        seen.add(key)
        covered.add(gid)
        out.append(row)
    if len(covered) != 1230:
        raise ValueError(f"Incomplete regular-season player coverage: {len(covered)}/1230 games")
    if len(out) < 10000:
        raise ValueError("Insufficient played regular-season player rows")
    return out


def load(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--games", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--out", default="research_data/regular_player_history")
    a = p.parse_args()
    source = load(a.stats)
    games = load(a.games)
    rows = filter_regular(games, source, a.season)
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"player_stats_regular_{a.season}.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    report = {"season": a.season, "source_rows": len(source),
              "regular_games": len({r["game"]["id"] for r in rows}),
              "played_regular_rows": len(rows),
              "excluded_rows": len(source)-len(rows),
              "rule": "Verified regular game IDs and complete 1230-game coverage, played minutes > 0"}
    (folder / f"manifest_{a.season}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
