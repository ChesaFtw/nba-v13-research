"""Fetch a complete season game index for fail-closed regular-season filtering."""
import argparse
import json
import os
import time
from pathlib import Path
import requests

URL = "https://api.balldontlie.io/v1/games"

def fetch_games(season, session=None, sleep=time.sleep):
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        raise RuntimeError("BALLDONTLIE_API_KEY is required")
    client = session or requests.Session()
    cursor, seen, games = None, set(), []
    for page in range(1, 31):
        params = {"seasons[]": season, "per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        for attempt in range(8):
            response = client.get(URL, headers={"Authorization": key}, params=params, timeout=35)
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == 7:
                    response.raise_for_status()
                raw = response.headers.get("Retry-After", "")
                try:
                    delay = max(1, min(float(raw), 120))
                except ValueError:
                    delay = min(10 * (attempt + 1), 120)
                sleep(delay)
                continue
            response.raise_for_status()
            break
        payload = response.json()
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ValueError("Malformed games response")
        games.extend(rows)
        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if next_cursor is None:
            return games
        if str(next_cursor) in seen:
            raise ValueError("Repeated games cursor")
        seen.add(str(next_cursor))
        cursor = next_cursor
        sleep(0.5)
    raise ValueError("Game-index page cap reached; refusing incomplete season")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--out", default="research_data/game_history")
    args = parser.parse_args()
    games = fetch_games(args.season)
    regular = [g for g in games if g.get("season") == args.season and g.get("postseason") is False]
    ids = [g.get("id") for g in regular]
    if len(regular) != 1230 or len(set(ids)) != 1230 or None in ids:
        raise ValueError(f"Expected 1230 unique regular games, got {len(regular)}")
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"games_{args.season}.jsonl").open("w") as output:
        for game in regular:
            output.write(json.dumps(game, separators=(",", ":")) + "\n")
    print(json.dumps({"season": args.season, "regular_games": len(regular), "source_games": len(games)}))

if __name__ == "__main__":
    main()
