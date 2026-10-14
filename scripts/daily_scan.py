"""Schedule inventory only: does NOT generate picks."""
from datetime import datetime, timezone
import json
from v13.bdl import get_games

def main() -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    games = get_games([today])
    output = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "requested_date_utc": today,
        "games_found": len(games),
        "games": [
            {"id": g.get("id"), "status": g.get("status"),
             "home_team": (g.get("home_team") or {}).get("abbreviation"),
             "visitor_team": (g.get("visitor_team") or {}).get("abbreviation")}
            for g in games
        ],
        "model_status": "DATA_INGESTION_ONLY_NO_VALIDATED_PICKS",
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
