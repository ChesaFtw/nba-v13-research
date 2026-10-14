"""Historical NBA regular-season game ingestion with cursor pagination and retries."""
import os
import time
import requests

URL = "https://api.balldontlie.io/v1/games"

def fetch_season(season: int, session=None) -> list[dict]:
    if not 1990 <= season <= 2100:
        raise ValueError("Invalid season")
    key = os.getenv("BALLDONTLIE_API_KEY")
    if not key:
        raise RuntimeError("BALLDONTLIE_API_KEY secret is missing")
    client = session or requests.Session()
    cursor = None
    seen_cursors = set()
    games = {}
    while True:
        params = {"seasons[]": season, "season_type": "regular", "per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        for attempt in range(5):
            response = client.get(URL, headers={"Authorization": key}, params=params, timeout=30)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(min(2 ** attempt, 8))
                continue
            response.raise_for_status()
            break
        payload = response.json()
        for game in payload.get("data", []):
            if game.get("id") is not None:
                games[str(game["id"])] = game
        cursor = payload.get("meta", {}).get("next_cursor")
        if cursor is None:
            break
        if str(cursor) in seen_cursors:
            raise RuntimeError(f"Repeated pagination cursor in season {season}")
        seen_cursors.add(str(cursor))
    return sorted(games.values(), key=lambda g: (str(g.get("date", "")), str(g.get("id"))))
