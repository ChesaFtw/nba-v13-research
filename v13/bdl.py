"""BALLDONTLIE NBA games client; API key comes only from environment."""
import os
import requests

BASE_URL = "https://api.balldontlie.io/v1"
VALID_SEASON_TYPES = {"preseason", "regular", "ist", "playin", "playoffs"}

def get_games(dates: list[str], season_type: str | None = None) -> list[dict]:
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        raise RuntimeError("Missing BALLDONTLIE_API_KEY GitHub Actions secret")
    if not dates:
        raise ValueError("At least one ISO date is required")
    if season_type is not None and season_type not in VALID_SEASON_TYPES:
        raise ValueError(f"Unknown season type: {season_type}")
    games = []
    cursor = None
    while True:
        params = [("dates[]", day) for day in dates] + [("per_page", 100)]
        if season_type:
            params.append(("season_type", season_type))
        if cursor is not None:
            params.append(("cursor", cursor))
        response = requests.get(
            f"{BASE_URL}/games",
            headers={"Authorization": key},
            params=params,
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()
        games.extend(payload.get("data", []))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            return games
