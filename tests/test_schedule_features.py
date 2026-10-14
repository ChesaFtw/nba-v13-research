from v13.schedule_features import schedule_features

def game(day, gid):
    return {"id": gid, "date": day, "home_team": {"id": 1}, "visitor_team": {"id": 2}}

def test_rest_and_back_to_back():
    rows = schedule_features([game("2025-10-24", 2), game("2025-10-23", 1)])
    assert rows[0]["home_rest_days"] is None
    assert rows[1]["home_rest_days"] == 0
    assert rows[1]["away_back_to_back"] is True

def test_multiple_days_rest():
    rows = schedule_features([game("2025-10-21", 1), game("2025-10-25", 2)])
    assert rows[1]["home_rest_days"] == 3
