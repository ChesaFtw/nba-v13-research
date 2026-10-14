"""Tests for exact regular-season filtering and DNP removal."""
import pytest
from scripts.regular_season_filter import filter_regular, minutes


def test_minutes_formats():
    assert minutes("12:30") == 12.5
    assert minutes("0:00") == 0
    assert minutes(None) == 0
    assert minutes("18") == 18


def test_regular_only_and_dnp_excluded():
    games = [{"id": i, "date": "2025-12-01", "season": 2025,
              "postseason": False} for i in range(1230)]
    stats = [{"game": {"id": i, "date": "2025-12-01"},
              "player": {"id": j}, "min": "12:30", "pts": 2}
             for i in range(1230) for j in range(10)]
    stats += [
        {"game": {"id": 1231, "date": "2025-12-01"},
         "player": {"id": 1}, "min": "30:00"},
        {"game": {"id": 0, "date": "2025-12-01"},
         "player": {"id": 99}, "min": "0:00"},
    ]
    cleaned = filter_regular(games, stats, 2025)
    assert len(cleaned) == 12300
    assert all(x["game"]["id"] < 1230 for x in cleaned)


def test_incomplete_game_set_fails_closed():
    with pytest.raises(ValueError, match="1230"):
        filter_regular([], [], 2025)


def test_duplicate_game_index_fails_closed():
    games = [{"id": i, "date": "2025-12-01", "season": 2025,
              "postseason": False} for i in range(1230)]
    with pytest.raises(ValueError, match="Duplicate regular-season game ID"):
        filter_regular(games + [games[0]], [], 2025)


def test_missing_played_game_fails_closed():
    games = [{"id": i, "date": "2025-12-01", "season": 2025,
              "postseason": False} for i in range(1230)]
    stats = [{"game": {"id": i, "date": "2025-12-01"},
              "player": {"id": j}, "min": "10:00"}
             for i in range(1229) for j in range(10)]
    with pytest.raises(ValueError, match="1229/1230 games"):
        filter_regular(games, stats, 2025)


def test_conflicting_player_game_season_fails_closed():
    games = [{"id": i, "date": "2025-12-01", "season": 2025,
              "postseason": False} for i in range(1230)]
    stats = [{"game": {"id": i, "date": "2025-12-01", "season": 2024},
              "player": {"id": j}, "min": "10:00"}
             for i in range(1230) for j in range(10)]
    with pytest.raises(ValueError, match="Player-game season mismatch"):
        filter_regular(games, stats, 2025)
