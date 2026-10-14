from v13.prop_features import build_prop_features, MARKETS

def row(day, pts, gid):
    return {"id": gid, "player": {"id": 1}, "game": {"id": gid, "date": day}, "pts": pts, "reb": 3, "ast": 2, "fg3m": 1}

def test_prop_features_exclude_current_game():
    rows = [row(f"2025-10-{d:02d}", d, d) for d in range(1, 7)]
    features = list(build_prop_features(list(reversed(rows))))
    assert len(features) == 1
    assert features[0]["points_avg_5"] == 3
    assert features[0]["points_actual"] == 6
    assert features[0]["pra_avg_5"] == 8
    assert set(MARKETS) == {"points", "rebounds", "assists", "threes", "pra"}

def test_same_day_games_do_not_leak():
    rows = [row(f"2025-10-{d:02d}", d, d) for d in range(1, 6)]
    rows += [row("2025-10-06", 100, 6), row("2025-10-06", 200, 7)]
    features = list(build_prop_features(rows))
    assert len(features) == 2
    assert all(f["points_avg_5"] == 3 for f in features)
