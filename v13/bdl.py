"""BALLDONTLIE API client. Key must be provided via environment variable."""
import os
import requests

BASE_URL = "https://api.balldontlie.io/v1"

def get_games(dates: list[str]) -> list[dict]:
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        raise RuntimeError("Set BALLDONTLIE_API_KEY as a GitHub Actions secret")
    if not dates:
        raise ValueError("At least one ISO date is required")
    games = []
    cursor = None
    while True:
        params = [("dates[]", day) for day in dates] + [("per_page", 100)]
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
