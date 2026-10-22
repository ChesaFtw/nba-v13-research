"""Guard against participant leakage and unjustified bankroll decisions."""
from research.pregame_approval_gate import review, american_break_even
from scripts.build_prior_only_game_features import build_prior_only


def _team(tri, player):
    return {
        "tricode": tri, "score": 110, "estimated_possessions": 99.,
        "effective_fg_pct": .51, "free_throw_rate": .24,
        "turnover_rate_est": .12,
        "four_factor_raw": {"field_goals_attempted": 80, "three_pointers_attempted": 30,
                            "offensive_rebounds": 10, "free_throws_attempted": 20},
        "statistics": {"pointsInPaint": "48", "fastBreakPoints": "12", "fouls": "19"},
        "players": [{"player_id": player, "played": True, "starter": True,
                     "points": 22, "minutes": 30, "reboundsTotal": 8,
                     "assists": 4, "fieldGoalsAttempted": 13}],
    }


def _game(gid, d, away, home):
    return {"game_id": gid, "game_date": d, "game_status": "final",
            "away": away, "home": home, "total": away["score"] + home["score"],
            "home_margin": home["score"] - away["score"]}


def test_future_game_roster_not_treated_as_known():
    first = _game("1", "2026-10-20", _team("MIL", "previous"), _team("WAS", "w"))
    second = _game("2", "2026-10-21", _team("MIL", "unknown_new_player"), _team("WAS", "w"))
    x = build_prior_only([first, second])
    assert x[0]["previous_roster_players"] == {}
    assert "previous" in x[1]["previous_roster_players"]
    assert "unknown_new_player" not in x[1]["previous_roster_players"]
    assert x[1]["teams"]["MIL"]["prior_game_count"] == 1
    assert x[1]["teams"]["MIL"]["prior5"]["score"] == 110


def test_same_date_results_not_used_for_second_game():
    g1 = _game("1", "2026-10-20", _team("A", "a"), _team("B", "b"))
    g2 = _game("2", "2026-10-20", _team("A", "z"), _team("C", "c"))
    rows = build_prior_only([g2, g1])
    assert all(row["teams"]["A"]["prior_game_count"] == 0 for row in rows)


def _valid_gate_row():
    return {
        "game_id": "401909835", "market": "total", "selection": "Under",
        "odds": -110, "quote_at": "2026-10-21T17:00:00Z",
        "tipoff_at": "2026-10-21T23:00:00Z",
        "injuries_at": "2026-10-21T16:30:00Z",
        "projected_lineup_at": "2026-10-21T16:00:00Z",
        "features_asof": "2026-10-20T23:00:00Z",
        "train_cutoff": "2026-10-20T00:00:00Z",
        "training_games": 2800, "holdout_games": 450,
        "calibration_mae": .032, "p_lower": .60,
        "probability_calibrated": True, "lineup_confidence": .91,
        "new_rotation_share": .15, "model_version": "hypothetical_test_only",
    }


def test_market_gate_fails_on_missing_model_validation():
    row = _valid_gate_row()
    row["probability_calibrated"] = False
    row["training_games"] = 8
    r = review(row, now="2026-10-21T17:10:00Z")
    assert not r["approved"]
    assert "too_few_training_games" in r["reasons"]
    assert "uncalibrated_probability" in r["reasons"]


def test_gate_rejects_roster_churn_without_robust_adjustment():
    row = _valid_gate_row()
    row["new_rotation_share"] = .42
    r = review(row, now="2026-10-21T17:10:00Z")
    assert "extreme_roster_churn_no_certified_adjustment" in r["reasons"]


def test_gate_rejects_betting_against_knicks():
    row = _valid_gate_row()
    row["market"] = "spread"
    row["opponent"] = "NYK"
    r = review(row, now="2026-10-21T17:10:00Z")
    assert "knicks_opponent_betting_rule" in r["reasons"]


def test_fair_american_break_even_and_hypothetical_approval():
    assert round(american_break_even(-110), 4) == .5238
    row = _valid_gate_row()
    assert review(row, now="2026-10-21T17:10:00Z")["approved"]
