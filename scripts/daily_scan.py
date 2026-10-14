"""NBA schedule diagnostic. No predictions or betting recommendations."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import json
from v13.bdl import get_games

CENTRAL = ZoneInfo("America/Chicago")
HISTORICAL_CONTROL = "2024-10-22"  # Known NBA regular-season opener

def summarize(games):
    return [
        {"id": g.get("id"), "date": g.get("date"), "status": g.get("status"),
         "home_team": (g.get("home_team") or {}).get("abbreviation"),
         "visitor_team": (g.get("visitor_team") or {}).get("abbreviation")}
        for g in games
    ]

def main():
    now = datetime.now(timezone.utc)
    central_day = now.astimezone(CENTRAL).date()
    dates = [(central_day + timedelta(days=offset)).isoformat() for offset in (0, 1)]
    checks = {}
    for season_type in ("preseason", "regular"):
        games = get_games(dates, season_type=season_type)
        checks[season_type] = {"count": len(games), "games": summarize(games)}
    control = get_games([HISTORICAL_CONTROL], season_type="regular")
    output = {
        "as_of_utc": now.isoformat(),
        "dates_central": dates,
        "checks": checks,
        "historical_control": {
            "date": HISTORICAL_CONTROL,
            "season_type": "regular",
            "count": len(control),
            "games": summarize(control),
        },
        "model_status": "DIAGNOSTIC_ONLY_NOT_A_VALIDATED_BETTING_MODEL",
    }
    print(json.dumps(output, indent=2))
    if not control:
        raise RuntimeError("Historical control returned zero games: investigate API coverage or parameters")

if __name__ == "__main__":
    main()
