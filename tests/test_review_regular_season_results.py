"""Tests for strict regular-season-only model promotion gates."""
import copy
import pytest
from scripts.review_regular_season_results import summarize, MARKETS, SEASONS

def fixtures():
    return {
        s: {"season": s, "regular_games": 1230, "holdout_rows": 1000,
            "markets": {
                m: {"rolling_holdout_mae": 3.0, "ridge_holdout_mae": 2.8,
                    "ridge_beats_selected_rolling": True}
                for m in MARKETS}}
        for s in SEASONS
    }

def test_four_season_success():
    result = summarize(fixtures())
    assert result["markets"]["points"]["holdout_rows"] == 4000
    assert result["markets"]["points"]["promote_ridge"] is True

def test_missing_season_rejected():
    reports = fixtures()
    reports.pop(2025)
    with pytest.raises(ValueError, match="four seasons"):
        summarize(reports)

def test_incomplete_regular_games_rejected():
    reports = fixtures()
    reports[2023]["regular_games"] = 1229
    with pytest.raises(ValueError, match="coverage"):
        summarize(reports)

def test_one_good_season_not_enough_to_promote():
    reports = fixtures()
    for season in (2023, 2024, 2025):
        for market in MARKETS:
            reports[season]["markets"][market]["ridge_holdout_mae"] = 3.1
            reports[season]["markets"][market]["ridge_beats_selected_rolling"] = False
    result = summarize(reports)
    assert result["markets"]["points"]["ridge_season_wins"] == 1
    assert result["markets"]["points"]["promote_ridge"] is False

def test_missing_market_rejected():
    reports = fixtures()
    del reports[2022]["markets"]["assists"]
    with pytest.raises(ValueError, match="markets"):
        summarize(reports)
