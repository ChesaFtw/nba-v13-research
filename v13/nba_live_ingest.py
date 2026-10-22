"""Official NBA postgame ingestion and diagnostics. No pregame predictions.

Source: https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json
        https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gameId}.json
        https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gameId}.json

Historical games must be FINAL; raw action data are archived unmodified.
"""
from __future__ import annotations
from datetime import datetime, timezone
import math
import re
import time
from typing import Any
import requests

BASE = "https://cdn.nba.com/static/json"
SCHEDULE_URL = f"{BASE}/staticData/scheduleLeagueV2.json"
GAME_ID_RE = re.compile(r"^002\d{7}$")
FIELDS = (
    "points", "minutes", "assists", "reboundsTotal", "reboundsOffensive",
    "reboundsDefensive", "fieldGoalsMade", "fieldGoalsAttempted",
    "threePointersMade", "threePointersAttempted", "freeThrowsMade",
    "freeThrowsAttempted", "turnovers", "steals", "blocks",
    "foulsPersonal", "plusMinusPoints", "pointsFastBreak",
    "pointsInThePaint", "pointsSecondChance",
)


class GameDataError(ValueError):
    pass


def fetch_json(url: str, timeout: int = 25, tries: int = 3) -> dict:
    if not url.startswith("https://cdn.nba.com/static/json/"):
        raise GameDataError("External URL not on approved NBA CDN")
    headers = {"User-Agent": "Mozilla/5.0 V13Research/1.0",
               "Accept": "application/json",
               "Referer": "https://www.nba.com/"}
    last_exc = None
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise GameDataError("Unexpected NBA JSON shape")
            return data
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt < tries - 1:
                time.sleep(2 ** attempt)
    raise GameDataError(f"Could not fetch {url}: {last_exc}")


def schedule_games(data: dict, dates: set[str]) -> list[dict]:
    """League schedule covers old dates unlike today's live scoreboard."""
    blocks = (data.get("leagueSchedule") or {}).get("gameDates") or []
    games = []
    seen = set()
    for block in blocks:
        for g in block.get("games") or []:
            gid = str(g.get("gameId", ""))
            code = str(g.get("gameCode", ""))
            date = (code[:8] if len(code) >= 8 else "").strip()
            if not GAME_ID_RE.fullmatch(gid) or date not in {d.replace("-", "") for d in dates}:
                continue
            if gid not in seen:
                seen.add(gid)
                games.append({"game_id": gid, "game_date": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                              "game_code": code})
    return sorted(games, key=lambda g: (g["game_date"], g["game_id"]))


def endpoints(game_id: str) -> tuple[str, str]:
    if not GAME_ID_RE.fullmatch(game_id):
        raise GameDataError(f"Invalid regular-season game ID: {game_id}")
    return (f"{BASE}/liveData/boxscore/boxscore_{game_id}.json",
            f"{BASE}/liveData/playbyplay/playbyplay_{game_id}.json")


def _value(stats: dict, key: str) -> float:
    value = stats.get(key) or 0
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise GameDataError(f"Non-numeric {key}: {value}")
    return float(value)


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 5) if den else None


def _normalize_team(team: dict, role: str) -> dict:
    stats = team.get("statistics") or {}
    players = []
    for p in team.get("players") or []:
        ps = p.get("statistics") or {}
        row = {"player_id": str(p.get("personId", "")), "name": p.get("name"),
               "team": team.get("teamTricode"), "starter": str(p.get("starter", "0")) == "1",
               "played": str(p.get("played", "0")) == "1",
               "status": p.get("status"), "not_playing_reason": p.get("notPlayingReason")}
        row.update({k: ps.get(k) for k in FIELDS})
        players.append(row)
    fga = _value(stats, "fieldGoalsAttempted")
    fgm = _value(stats, "fieldGoalsMade")
    threes = _value(stats, "threePointersMade")
    fta = _value(stats, "freeThrowsAttempted")
    ftm = _value(stats, "freeThrowsMade")
    oreb = _value(stats, "reboundsOffensive")
    tov = _value(stats, "turnoversTotal") or _value(stats, "turnovers")
    pts = float(team.get("score", 0))
    if pts < 0 or fga < 1:
        raise GameDataError(f"Invalid completed-game team score/FGA: {role}")
    if abs((2 * fgm + threes + ftm) - pts) > 1:
        raise GameDataError(f"Score not reconciled for {role}")
    est_poss = fga - oreb + tov + 0.44 * fta
    if est_poss <= 0:
        raise GameDataError("Nonpositive estimated possessions")
    return {
        "role": role, "team_id": str(team.get("teamId")),
        "tricode": team.get("teamTricode"), "score": int(pts),
        "periods": team.get("periods") or [],
        "statistics": stats, "players": players,
        "estimated_possessions": round(est_poss, 2),
        "effective_fg_pct": _ratio(fgm + 0.5 * threes, fga),
        "free_throw_rate": _ratio(fta, fga),
        "turnover_rate_est": _ratio(tov, est_poss),
    }


def _quarter_scores(team: dict) -> list[int]:
    periods = team.get("periods") or []
    scores = []
    for p in periods:
        if p.get("periodType") in (None, "REGULAR"):
            scores.append(int(p.get("score") or 0))
    return scores


def normalize_game(box: dict, play: dict, game_id: str, game_date: str) -> dict:
    bg = box.get("game") or {}
    pg = play.get("game") or {}
    if str(bg.get("gameId")) != game_id or str(pg.get("gameId")) != game_id:
        raise GameDataError("Game IDs do not agree between boxscore and PBP")
    if bg.get("gameStatus") != 3:
        raise GameDataError("Game not final; refusing to grade or train")
    away = _normalize_team(bg.get("awayTeam") or {}, "away")
    home = _normalize_team(bg.get("homeTeam") or {}, "home")
    actions = pg.get("actions")
    if not isinstance(actions, list) or len(actions) < 150:
        raise GameDataError("Missing/incomplete play-by-play")
    final_scores = [(a.get("scoreAway"), a.get("scoreHome")) for a in actions
                    if str(a.get("scoreAway", "")).strip() and str(a.get("scoreHome", "")).strip()]
    if not final_scores or tuple(map(int, final_scores[-1])) != (away["score"], home["score"]):
        raise GameDataError("Last play-by-play score does not reconcile with boxscore")
    a_q = _quarter_scores(bg["awayTeam"])
    h_q = _quarter_scores(bg["homeTeam"])
    if sum(a_q) != away["score"] or sum(h_q) != home["score"]:
        raise GameDataError("Period scores fail sum-to-final check")
    team_poss = (away["estimated_possessions"] + home["estimated_possessions"]) / 2
    game_minutes = 48 + max(0, len(a_q) - 4) * 5
    if team_poss <= 0:
        raise GameDataError("Invalid possessions")
    periods = [dict(period=i + 1, away=a_q[i], home=h_q[i], total=a_q[i] + h_q[i])
               for i in range(len(a_q))]
    subevents = []
    action_types = {}
    shot_attempts = 0
    for a in actions:
        kind = str(a.get("actionType") or "").lower()
        action_types[kind] = action_types.get(kind, 0) + 1
        shot_attempts += 1 if a.get("isFieldGoal") in (1, True) else 0
        if kind == "substitution":
            subevents.append({
                "period": a.get("period"), "clock": a.get("clock"),
                "team": a.get("teamTricode"), "person_id": a.get("personId"),
                "sub_type": a.get("subType"), "action_number": a.get("actionNumber"),
                "description": a.get("description")})
    result = {
        "game_id": game_id, "game_date": game_date,
        "game_status": "final",
        "source": "NBA CDN boxscore + playbyplay",
        "source_urls": list(endpoints(game_id)),
        "away": away, "home": home,
        "total": home["score"] + away["score"],
        "home_margin": home["score"] - away["score"],
        "period_scores": periods,
        "estimated_game_possessions": round(team_poss, 2),
        "estimated_pace_48": round(team_poss * 48 / game_minutes, 2),
        "away_off_rating_est": round(away["score"] * 100 / team_poss, 2),
        "home_off_rating_est": round(home["score"] * 100 / team_poss, 2),
        "playbyplay_action_count": len(actions),
        "playbyplay_field_goal_events": shot_attempts,
        "playbyplay_event_counts": action_types,
        "substitutions": subevents,
        "quality": {
            "final_box_score": True, "scores_reconciled": True,
            "playbyplay_complete_score": True,
            "possessions_are_boxscore_estimates": True,
            "lineup_stints_validated": False,
            "opponent_defender_assignments_available": False,
        },
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return result
