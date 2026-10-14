"""Filter player box scores using verified regular-season game IDs."""
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
    valid = {str(g["id"]): str(g["date"])[:10] for g in games
             if g.get("season") == season and g.get("postseason") is False}
    if len(valid) != 1230:
        raise ValueError(f"Expected 1230 regular games; found {len(valid)}")
    out, seen = [], set()
    for r in stats:
        game = r.get("game") or {}
        gid = str(game.get("id"))
        if gid not in valid or minutes(r.get("min")) <= 0:
            continue
        if str(game.get("date", ""))[:10] != valid[gid]:
            raise ValueError("Player-game date mismatch")
        pid = (r.get("player") or {}).get("id")
        if pid is None:
            continue
        key = (gid, pid)
        if key in seen:
            raise ValueError("Duplicate player-game")
        seen.add(key)
        out.append(r)
    if len(out) < 10000 or len({r["game"]["id"] for r in out}) < 1200:
        raise ValueError("Incomplete regular-season player coverage")
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
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    report = {"season": a.season, "source_rows": len(source),
              "regular_games": len(games), "played_regular_rows": len(rows),
              "excluded_rows": len(source)-len(rows),
              "rule": "Verified regular game IDs, played minutes > 0"}
    (folder / f"manifest_{a.season}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
