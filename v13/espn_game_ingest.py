"""ESPN public NBA feed -> validated game/player/possession feature archive.

Source: site.api.espn.com/apis/site/v2/sports/basketball/nba/{scoreboard,summary}
NBA's CDN currently responds HTTP 403 in hosted Actions, so this is the live
fallback. ESPN play-by-play has event descriptions, players, shot coordinates,
game clock and score; it does NOT identify defender assignments.
"""
from __future__ import annotations
from datetime import datetime, timezone
from math import isfinite
from typing import Any
import requests
import time

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
NUMERIC_KEYS = ("points", "assists", "rebounds", "offensiveRebounds",
                "defensiveRebounds", "turnovers", "steals", "blocks",
                "fouls", "plusMinus", "minutes")
PAIRS = {
    "fieldGoalsMade-fieldGoalsAttempted": ("fieldGoalsMade", "fieldGoalsAttempted"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("threePointersMade", "threePointersAttempted"),
    "freeThrowsMade-freeThrowsAttempted": ("freeThrowsMade", "freeThrowsAttempted"),
}


class ESPNDataError(ValueError):
    pass


def get_json(url: str, tries: int = 3) -> dict:
    if not url.startswith(BASE + "/") or not ("scoreboard?" in url or "summary?" in url):
        raise ESPNDataError("Disallowed ESPN source URL")
    err = None
    for retry in range(tries):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0",
                         "Accept": "application/json"}, timeout=25)
            response.raise_for_status()
            obj = response.json()
            if not isinstance(obj, dict):
                raise ESPNDataError("Nonobject response")
            return obj
        except (ValueError, requests.RequestException) as exc:
            err = exc
            if retry < tries - 1:
                time.sleep(2 ** retry)
    raise ESPNDataError(f"ESPN HTTP/JSON failure: {err}")


def scoreboard_url(day: str) -> str:
    if len(day) != 10 or day[4] != "-" or day[7] != "-":
        raise ESPNDataError("Date must be YYYY-MM-DD")
    return f"{BASE}/scoreboard?dates={day.replace('-', '')}&limit=100"


def summary_url(event_id: str) -> str:
    if not str(event_id).isdigit():
        raise ESPNDataError("Invalid ESPN event id")
    return f"{BASE}/summary?event={event_id}"


def regular_final_events(board: dict) -> list[dict]:
    answer = []
    for e in board.get("events") or []:
        comp = (e.get("competitions") or [{}])[0]
        complete = ((e.get("status") or {}).get("type") or {}).get("completed")
        season = (e.get("season") or {}).get("type")
        # Only regular season, NEVER preseason/postseason or unfinished games.
        if complete is True and season == 2 and len(comp.get("competitors") or []) == 2:
            answer.append(e)
    return answer


def _number(value, default=0.0):
    if value is None or str(value).strip() in ("", "--", "-", "DNP"):
        return float(default)
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ESPNDataError(f"Non-numeric box score field: {str(value)[:40]}")
    if not isfinite(f):
        raise ESPNDataError("Nonfinite box score field")
    return f


def _split_pair(raw):
    a = str(raw).split("-")
    if len(a) != 2:
        raise ESPNDataError(f"Invalid made-attempted pair: {raw}")
    return int(_number(a[0])), int(_number(a[1]))


def _minutes(raw):
    if raw in (None, "", "--"):
        return 0.0
    val = str(raw)
    if ":" in val:
        x, y = val.split(":", 1)
        return round(_number(x) + _number(y) / 60, 3)
    return _number(val)


def _stats_index(items: list) -> dict:
    return {str(s.get("name")): str(s.get("displayValue") or "")
            for s in items if s.get("name")}


def _team(box_entry, competitor, role):
    stats = _stats_index(box_entry.get("statistics") or [])
    tri = competitor["team"]["abbreviation"]
    score = int(_number(competitor.get("score")))
    fgm, fga = _split_pair(stats["fieldGoalsMade-fieldGoalsAttempted"])
    three, threea = _split_pair(stats["threePointFieldGoalsMade-threePointFieldGoalsAttempted"])
    ftm, fta = _split_pair(stats["freeThrowsMade-freeThrowsAttempted"])
    if 2 * fgm + three + ftm != score:
        raise ESPNDataError(f"Score breakdown mismatch for {tri}")
    to = _number(stats.get("totalTurnovers") or stats.get("turnovers"))
    orb = _number(stats.get("offensiveRebounds"))
    poss = fga - orb + to + 0.44 * fta
    if poss <= 0 or fga <= 0:
        raise ESPNDataError("Estimated possessions invalid")
    return {"role": role, "team_id": str(competitor["team"]["id"]),
            "tricode": tri, "score": score, "statistics": stats,
            "estimated_possessions": round(poss, 3),
            "effective_fg_pct": round((fgm + .5 * three) / fga, 5),
            "free_throw_rate": round(fta / fga, 5),
            "turnover_rate_est": round(to / poss, 5),
            "four_factor_raw": {
                "field_goals_made": fgm, "field_goals_attempted": fga,
                "three_pointers_made": three, "three_pointers_attempted": threea,
                "free_throws_made": ftm, "free_throws_attempted": fta,
                "offensive_rebounds": orb, "turnovers": to},
            "players": []}


def _players(section: dict):
    out = []
    for group in section.get("statistics") or []:
        keys = group.get("keys") or []
        for row in group.get("athletes") or []:
            athlete = row.get("athlete") or {}
            vals = dict(zip(keys, row.get("stats") or []))
            obj = {
                "player_id": str(athlete.get("id", "")),
                "name": athlete.get("displayName"),
                "team": (section.get("team") or {}).get("abbreviation"),
                "position": (athlete.get("position") or {}).get("abbreviation"),
                "starter": bool(row.get("starter")),
                "played": not bool(row.get("didNotPlay")) and bool(vals.get("minutes")),
                "did_not_play": bool(row.get("didNotPlay")),
                "not_playing_reason": row.get("reason"),
            }
            for k in NUMERIC_KEYS:
                obj[k] = _minutes(vals.get(k)) if k == "minutes" else _number(vals.get(k))
            for source, targets in PAIRS.items():
                made, attempted = _split_pair(vals[source]) if vals.get(source) else (0, 0)
                obj[targets[0]] = made
                obj[targets[1]] = attempted
            obj["reboundsTotal"] = obj["rebounds"]
            obj["reboundsOffensive"] = obj["offensiveRebounds"]
            obj["reboundsDefensive"] = obj["defensiveRebounds"]
            obj["plusMinusPoints"] = obj["plusMinus"]
            # Preserve ESPN IDs: these are not interchangeable with NBA person IDs.
            out.append(obj)
    return out


def _periods(plays: list) -> list[dict]:
    # Derive quarter scoring from the cumulative scoreboard carried by PBP.
    last_score = {}
    for a in plays:
        num = (a.get("period") or {}).get("number")
        if isinstance(num, int) and num > 0:
            away = a.get("awayScore")
            home = a.get("homeScore")
            if away is not None and home is not None:
                last_score[num] = (int(away), int(home))
    if not last_score or sorted(last_score) != list(range(1, max(last_score) + 1)):
        raise ESPNDataError("PBP lacks contiguous period scoring")
    result = []
    prev_away = prev_home = 0
    for period in sorted(last_score):
        away, home = last_score[period]
        a, h = away - prev_away, home - prev_home
        if a < 0 or h < 0:
            raise ESPNDataError("Period score decreased")
        result.append({"period": period, "away": a, "home": h, "total": a + h})
        prev_away, prev_home = away, home
    return result


def normalize(event: dict, summary: dict, day: str) -> dict:
    eid = str(event["id"])
    if not event.get("status", {}).get("type", {}).get("completed"):
        raise ESPNDataError("Not a finished game")
    if (event.get("season") or {}).get("type") != 2:
        raise ESPNDataError("Not a regular-season game")
    header = (summary.get("header") or {}).get("competitions") or []
    if not header or str(header[0].get("id")) != eid:
        raise ESPNDataError("ESPN summary game id disagrees with scoreboard")
    comps = (event.get("competitions") or [{}])[0].get("competitors") or []
    roster = {x.get("homeAway"): x for x in comps}
    if set(roster) != {"away", "home"}:
        raise ESPNDataError("Missing home/away competitors")
    boxteams = {(t.get("team") or {}).get("id"): t
                for t in (summary.get("boxscore") or {}).get("teams") or []}
    teams = {}
    for role in ("away", "home"):
        comp = roster[role]
        boxteam = boxteams.get(comp["team"]["id"])
        if boxteam is None:
            raise ESPNDataError(f"Missing team box score for {role}")
        teams[role] = _team(boxteam, comp, role)
    for section in (summary.get("boxscore") or {}).get("players") or []:
        role_team = (section.get("team") or {}).get("id")
        for role in ("away", "home"):
            if role_team == teams[role]["team_id"]:
                teams[role]["players"] = _players(section)
    if any(not teams[r]["players"] for r in ("away", "home")):
        raise ESPNDataError("Missing player statistics")
    plays = summary.get("plays") or []
    if len(plays) < 150:
        raise ESPNDataError("Too few PBP actions for a final NBA game")
    if int(plays[-1]["awayScore"]) != teams["away"]["score"] or \
       int(plays[-1]["homeScore"]) != teams["home"]["score"]:
        raise ESPNDataError("PBP final score does not match box scores")
    periods = _periods(plays)
    if (sum(x["away"] for x in periods), sum(x["home"] for x in periods)) != \
       (teams["away"]["score"], teams["home"]["score"]):
        raise ESPNDataError("PBP quarters do not sum to game total")
    poss = (teams["away"]["estimated_possessions"] +
            teams["home"]["estimated_possessions"]) / 2
    game_minutes = 48 + max(0, len(periods) - 4) * 5
    # ESPN PBP action type is free-text; retain original events in compressed raw.
    events = {}
    substitutions = []
    shots = []
    for play in plays:
        kind = str((play.get("type") or {}).get("text") or "unknown")
        events[kind] = events.get(kind, 0) + 1
        if "substitution" in kind.lower():
            substitutions.append({
                "period": (play.get("period") or {}).get("number"),
                "clock": (play.get("clock") or {}).get("displayValue"),
                "team_id": (play.get("team") or {}).get("id"),
                "description": play.get("text"), "participants": play.get("participants")})
        if play.get("shootingPlay"):
            shots.append({
                "period": (play.get("period") or {}).get("number"),
                "clock": (play.get("clock") or {}).get("displayValue"),
                "team_id": (play.get("team") or {}).get("id"),
                "coordinates": play.get("coordinate"), "scoring_play": play.get("scoringPlay"),
                "points_attempted": play.get("pointsAttempted")})
    return {
        "game_id": eid, "game_date": day, "game_status": "final",
        "source": "ESPN scoreboard and summary", "source_urls": [scoreboard_url(day), summary_url(eid)],
        "away": teams["away"], "home": teams["home"],
        "total": teams["away"]["score"] + teams["home"]["score"],
        "home_margin": teams["home"]["score"] - teams["away"]["score"],
        "period_scores": periods, "estimated_game_possessions": round(poss, 3),
        "estimated_pace_48": round(poss * 48 / game_minutes, 3),
        "away_off_rating_est": round(100 * teams["away"]["score"] / poss, 3),
        "home_off_rating_est": round(100 * teams["home"]["score"] / poss, 3),
        "playbyplay_action_count": len(plays),
        "playbyplay_event_counts": events,
        "substitutions": substitutions, "shot_events": shots,
        "quality": {
            "final_box_score": True, "scores_reconciled": True,
            "playbyplay_complete_score": True,
            "possessions_are_boxscore_estimates": True,
            "lineup_stints_validated": False,
            "opponent_defender_assignments_available": False,
        },
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
    }
