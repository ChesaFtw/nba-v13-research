"""Collect ALL opening player-prop rows (every prop type, two-sided only) for one season's games.

Same endpoint and filters as collect_full_prop_openings.py, but season-agnostic and with a
manifest of rows per prop type so coverage gaps are visible before any experiment runs.

python -m scripts.collect_season_prop_openings --games games_2024.jsonl --out research_data/props/opening_props_2024.jsonl
"""
import argparse, json, os, time
from collections import Counter
from pathlib import Path

import requests

URL = "https://api.balldontlie.io/nba/v2/odds/player_props/opening"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True); p.add_argument("--out", required=True)
    p.add_argument("--min-games", type=int, default=1100)
    a = p.parse_args()
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        raise RuntimeError("BALLDONTLIE_API_KEY not set")
    games = [json.loads(x) for x in Path(a.games).open() if x.strip()]
    games = sorted((g for g in games if g.get("status") == "Final" and g.get("postseason") is False),
                   key=lambda g: (str(g.get("date")), str(g.get("id"))))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    s = requests.Session(); counts = Counter(); by_type = Counter(); games_with_rows = 0
    with out.open("w") as f:
        for g in games:
            for attempt in range(7):
                r = s.get(URL, params={"game_id": g["id"]}, headers={"Authorization": key}, timeout=60)
                if r.status_code in (429, 500, 502, 503, 504):
                    if attempt == 6:
                        r.raise_for_status()
                    time.sleep(min(5 * (attempt + 1), 30)); continue
                r.raise_for_status(); break
            data = r.json().get("data", [])
            usable = 0
            for row in data:
                if str(row.get("game_id")) != str(g["id"]):
                    raise RuntimeError("Game ID mismatch")
                if (row.get("market") or {}).get("type") != "over_under" or not row.get("opened_at") or row.get("line_value") is None:
                    continue
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                usable += 1; by_type[str(row.get("prop_type"))] += 1
            counts["games"] += 1; counts["usable_rows"] += usable; games_with_rows += usable > 0
            if counts["games"] % 100 == 0:
                print(dict(counts), flush=True)
            time.sleep(0.06)
    manifest = {**counts, "games_with_rows": games_with_rows, "rows_by_prop_type": dict(by_type)}
    (out.parent / (out.stem + "_manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if counts["games"] < a.min_games:
        raise RuntimeError(f"Only {counts['games']} final regular-season games in index")


if __name__ == "__main__":
    main()
