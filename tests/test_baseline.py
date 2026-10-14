from v13.baseline import evaluate_seasons, summarize


def game(gid, date, h, a, hs, vs, status="Final"):
    return {"id": gid, "date": date, "home_team": {"id": h},
            "visitor_team": {"id": a}, "home_team_score": hs,
            "visitor_team_score": vs, "status": status}


def test_pregame_probability_not_affected_by_own_outcome():
    first = game(1, "2023-01-01", 1, 2, 110, 100)
    other = game(1, "2023-01-01", 1, 2, 90, 100)
    assert evaluate_seasons([(2022, [first])])[0]["p_home"] == evaluate_seasons([(2022, [other])])[0]["p_home"]


def test_chronological_update_and_future_exclusion():
    games = [game(2, "2023-01-02", 1, 2, 105, 90),
             game(1, "2023-01-01", 1, 2, 120, 90),
             game(3, "2023-01-03", 1, 2, 0, 0, "Scheduled")]
    result = evaluate_seasons([(2022, games)])
    assert len(result) == 2
    assert result[0]["game_id"] == 1
    assert result[1]["p_home"] > result[0]["p_home"]
    assert summarize(result)["games"] == 2


def test_season_regression_keeps_valid_probabilities():
    records = evaluate_seasons([(2022, [game(1, "2022-11-01", 1, 2, 110, 100)]),
                                (2023, [game(2, "2023-11-01", 1, 2, 99, 101)])])
    assert len(records) == 2
    assert all(0 < row["p_home"] < 1 for row in records)
