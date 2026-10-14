"""Point-in-time schedule features from historical game schedules."""
from collections import defaultdict
from datetime import date

def schedule_features(games):
    """Generate pregame rest features, independent of final scores."""
    previous = {}
    recent = defaultdict(list)
    output = []
    for g in sorted(games, key=lambda x: (str(x.get("date", "")), str(x.get("id", "")))):
        day = date.fromisoformat(str(g["date"])[:10])
        home = g.get("home_team") or {}
        away = g.get("visitor_team") or {}
        hid = home.get("id") if isinstance(home, dict) else home
        aid = away.get("id") if isinstance(away, dict) else away
        if hid is None or aid is None or hid == aid:
            continue
        values = {}
        for label, team in (("home", hid), ("away", aid)):
            last = previous.get(team)
            rest = (day - last).days - 1 if last else None
            values[f"{label}_rest_days"] = rest
            values[f"{label}_back_to_back"] = rest == 0
            values[f"{label}_games_last_4_days"] = sum(0 <= (day - d).days <= 3 for d in recent[team] if d < day)
        output.append({"game_id": g.get("id"), "date": str(day), **values})
        for team in (hid, aid):
            if team not in previous or day > previous[team]:
                previous[team] = day
            recent[team].append(day)
    return output
