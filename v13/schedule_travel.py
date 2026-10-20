"""Point-in-time rest, schedule-density and travel features for both teams.

Venue is assumed to be the home team's arena (neutral-site games such as
international series are rare and are not handled). Coordinates are arena
approximations; timezone offsets ignore daylight saving (only differences matter).
"""
import math
from datetime import date

# abbreviation: (lat, lon, utc_offset_standard, altitude_m)
ARENAS = {
    "ATL": (33.757, -84.396, -5, 320), "BOS": (42.366, -71.062, -5, 5), "BKN": (40.683, -73.975, -5, 10),
    "CHA": (35.225, -80.839, -5, 230), "CHI": (41.881, -87.674, -6, 180), "CLE": (41.497, -81.688, -5, 200),
    "DAL": (32.790, -96.810, -6, 130), "DEN": (39.749, -105.008, -7, 1609), "DET": (42.341, -83.055, -5, 190),
    "GSW": (37.768, -122.388, -8, 5), "HOU": (29.751, -95.362, -6, 15), "IND": (39.764, -86.155, -5, 220),
    "LAC": (33.945, -118.341, -8, 30), "LAL": (34.043, -118.267, -8, 80), "MEM": (35.138, -90.051, -6, 80),
    "MIA": (25.781, -80.188, -5, 2), "MIL": (43.045, -87.917, -6, 180), "MIN": (44.979, -93.276, -6, 260),
    "NOP": (29.949, -90.082, -6, 2), "NYK": (40.751, -73.993, -5, 10), "OKC": (35.463, -97.515, -6, 365),
    "ORL": (28.539, -81.384, -5, 30), "PHI": (39.901, -75.172, -5, 10), "PHX": (33.446, -112.071, -7, 330),
    "POR": (45.532, -122.667, -8, 15), "SAC": (38.580, -121.500, -8, 8), "SAS": (29.427, -98.437, -6, 200),
    "TOR": (43.643, -79.379, -5, 80), "UTA": (40.768, -111.901, -7, 1288), "WAS": (38.898, -77.021, -5, 10),
}
HIGH_ALTITUDE_M = 1000


def _km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def _abbr(team):
    return team.get("abbreviation") if isinstance(team, dict) else None


def _id(team):
    return str(team.get("id")) if isinstance(team, dict) else str(team)


def schedule_travel_features(games):
    """game_id -> features. Uses only the schedule (dates/venues), never results."""
    last = {}       # team -> (date, venue_abbr)
    history = {}    # team -> list of dates
    out = {}
    for g in sorted(games, key=lambda x: (str(x.get("date", "")), str(x.get("id", "")))):
        day = date.fromisoformat(str(g["date"])[:10])
        venue = _abbr(g.get("home_team"))
        feats = {}
        for side, team in (("home", g.get("home_team")), ("away", g.get("visitor_team"))):
            tid = _id(team)
            prev = last.get(tid)
            rest = (day - prev[0]).days - 1 if prev else None
            feats[f"{side}_rest_days"] = rest
            feats[f"{side}_b2b"] = int(rest == 0) if rest is not None else 0
            feats[f"{side}_games_last_4d"] = sum(0 < (day - d).days <= 3 for d in history.get(tid, []))
            km = tz = None
            if prev and venue in ARENAS and prev[1] in ARENAS:
                km = _km(ARENAS[prev[1]], ARENAS[venue])
                tz = ARENAS[venue][2] - ARENAS[prev[1]][2]
            feats[f"{side}_travel_km"] = km
            feats[f"{side}_tz_shift"] = tz
        feats["away_at_altitude"] = int(venue in ARENAS and ARENAS[venue][3] >= HIGH_ALTITUDE_M)
        r_h, r_a = feats["home_rest_days"], feats["away_rest_days"]
        feats["rest_diff"] = (min(r_h, 3) - min(r_a, 3)) if r_h is not None and r_a is not None else 0
        out[str(g["id"])] = feats
        for team in (g.get("home_team"), g.get("visitor_team")):
            tid = _id(team)
            last[tid] = (day, venue)
            history.setdefault(tid, []).append(day)
    return out
