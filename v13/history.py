"""Historical NBA games downloader with respectful API pacing and bounded retries."""
import os
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import requests

URL = "https://api.balldontlie.io/v1/games"
PAGE_INTERVAL_SECONDS = float(os.getenv("BDL_PAGE_INTERVAL_SECONDS", "1.0"))
MAX_RETRIES = 12

def _retry_delay(response, attempt):
    raw = response.headers.get("Retry-After") if hasattr(response, "headers") else None
    if raw:
        try:
            return min(max(float(raw), 1), 180)
        except ValueError:
            try:
                delta = (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds()
                return min(max(delta, 1), 180)
            except (ValueError, TypeError, OverflowError):
                pass
    return min(10 * (attempt + 1), 180)

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
    page = 0
    while True:
        params = {"seasons[]": season, "season_type": "regular", "per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        for attempt in range(MAX_RETRIES + 1):
            response = client.get(URL, headers={"Authorization": key}, params=params, timeout=30)
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt >= MAX_RETRIES:
                    response.raise_for_status()
                delay = _retry_delay(response, attempt)
                print(f"Season {season} page {page + 1}: HTTP {response.status_code}; waiting {delay:.0f}s", flush=True)
                time.sleep(delay)
                continue
            response.raise_for_status()
            break
        payload = response.json()
        for game in payload.get("data", []):
            if game.get("id") is not None:
                games[str(game["id"])] = game
        page += 1
        if page % 5 == 0:
            print(f"Season {season}: {page} pages, {len(games)} unique games", flush=True)
        cursor = payload.get("meta", {}).get("next_cursor")
        if cursor is None:
            break
        if str(cursor) in seen_cursors:
            raise RuntimeError(f"Repeated pagination cursor in season {season}")
        seen_cursors.add(str(cursor))
        if session is None:
            time.sleep(PAGE_INTERVAL_SECONDS)
    return sorted(games.values(), key=lambda g: (str(g.get("date", "")), str(g.get("id"))))
