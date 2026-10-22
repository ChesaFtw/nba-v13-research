"""Offline reproducible contract tests; never depend on today's games."""
import copy
import pytest
from v13.nba_live_ingest import normalize_game, schedule_games, GameDataError


def fixture():
    def team(teamid, tri, score, pts, periods):
        # 80 2s, 10 3s, 8 FTs = 198: use a numerically coherent toy score instead
        return {
            "teamId": teamid, "teamTricode": tri, "score": score,
            "periods": [{"period": i + 1, "periodType": "REGULAR", "score": q}
                        for i, q in enumerate(periods)],
            "statistics": {"fieldGoalsAttempted": 80, "fieldGoalsMade": 40,
                           "threePointersMade": 10, "freeThrowsAttempted": 24,
                           "freeThrowsMade": score - 90,
                           "reboundsOffensive": 10, "turnoversTotal": 12},
            "players": [{"personId": teamid * 100, "name": "A. Player",
                         "starter": "1", "played": "1",
                         "statistics": {"points": score, "assists": 6, "plusMinusPoints": 5}}]
        }
    away = team(101, "MIL", 110, 110, [25, 25, 30, 30])
    home = team(102, "WAS", 120, 120, [28, 32, 30, 30])
    actions = [{"actionNumber": i, "period": 4, "clock": "PT0M0.0S",
                "actionType": "shot", "scoreAway": str(110 if i > 150 else 99),
                "scoreHome": str(120 if i > 150 else 95)}
               for i in range(200)]
    actions.append({"actionNumber": 202, "period": 4, "clock": "PT0M0.0S",
                    "actionType": "substitution", "subType": "in", "personId": 10100,
                    "teamTricode": "MIL", "description": "Player enters",
                    "scoreAway": "110", "scoreHome": "120"})
    box = {"game": {"gameId": "0022600086", "gameStatus": 3,
                    "awayTeam": away, "homeTeam": home}}
    pbp = {"game": {"gameId": "0022600086", "actions": actions}}
    return box, pbp


def test_normalizes_and_reconciles_final():
    b, p = fixture()
    r = normalize_game(b, p, "0022600086", "2026-10-21")
    assert r["total"] == 230
    assert r["home_margin"] == 10
    assert r["estimated_game_possessions"] == 92.56
    assert r["period_scores"][2]["total"] == 60
    assert r["away"]["players"][0]["starter"] is True
    assert r["substitutions"][0]["person_id"] == 10100
    assert r["quality"]["lineup_stints_validated"] is False


def test_prevents_premature_grade():
    b, p = fixture()
    b["game"]["gameStatus"] = 2
    with pytest.raises(GameDataError, match="not final"):
        normalize_game(b, p, "0022600086", "2026-10-21")


def test_rejects_pbp_score_disagreement():
    b, p = fixture()
    p["game"]["actions"][-1]["scoreHome"] = "119"
    with pytest.raises(GameDataError, match="does not reconcile"):
        normalize_game(b, p, "0022600086", "2026-10-21")


def test_rejects_untallied_score():
    b, p = fixture()
    b["game"]["homeTeam"]["statistics"]["freeThrowsMade"] = 7
    with pytest.raises(GameDataError, match="not reconciled"):
        normalize_game(b, p, "0022600086", "2026-10-21")


def test_schedule_dedupe_and_filter_by_date_and_season():
    schedule = {"leagueSchedule": {"gameDates": [
        {"games": [
            {"gameId": "0022600086", "gameCode": "20261021/MILWAS"},
            {"gameId": "0012600087", "gameCode": "20261021/MILWAS"},
            {"gameId": "0022600086", "gameCode": "20261021/MILWAS"},
            {"gameId": "0022600088", "gameCode": "20261022/PHXPOR"},
        ]}
    ]}}
    g = schedule_games(schedule, {"2026-10-21"})
    assert len(g) == 1 and g[0]["game_id"] == "0022600086"
