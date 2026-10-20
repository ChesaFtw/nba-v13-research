import math
import random
from datetime import date, timedelta

import pytest

from scripts.h1_blowout_minutes import run
from v13.provenance import UnverifiedOddsError, quarantine


def world(seed=0, book_knows_blowouts=False, n_days=150):
    """Two starters per team; in |spread|>=9.5 games their minutes drop ~7. Lines either ignore that or not."""
    rng = random.Random(seed)
    games, odds, props, stats = [], [], [], []
    teams = list(range(1, 11))
    gid = 0
    for d in range(n_days):
        day = (date(2025, 10, 22) + timedelta(days=d)).isoformat()
        rng.shuffle(teams)
        for i in range(0, 10, 2):
            h, a = teams[i], teams[i + 1]
            gid += 1
            spread = rng.choice([-14.5, -11.5, -9.5, -6.5, -3.5, -1.5, 2.5, 4.5, 8.5, 12.5])  # home handicap
            games.append({"id": gid, "date": day, "datetime": day + "T23:30:00Z", "season": 2025,
                          "home_team": {"id": h}, "visitor_team": {"id": a}, "status": "Final", "postseason": False})
            odds.append({"game_id": gid, "vendor": "draftkings", "opened_at": day + "T14:00:00Z",
                         "spread_home_value": str(spread), "spread_home_odds": -110, "spread_away_odds": -110,
                         "total_value": "225.5", "total_over_odds": -110, "total_under_odds": -110})
            blow = abs(spread) >= 9.5
            for team in (h, a):
                for k in range(2):
                    pid = team * 10 + k
                    mins = max(0.0, rng.gauss(34 - (7 if blow else 0), 3))
                    rate = 0.7 + 0.1 * k
                    pts = sum(rng.random() < rate for _ in range(int(mins)))
                    stats.append({"min": f"{int(mins)}:00", "pts": pts, "reb": 0, "ast": 0, "fg3m": 0,
                                  "player": {"id": pid}, "team": {"id": team}, "game": {"id": gid, "date": day}})
                    exp_mins = 34 - (7 if (blow and book_knows_blowouts) else 0)
                    line = math.floor(exp_mins * rate) + 0.5
                    props.append({"game_id": gid, "player_id": pid, "vendor": "draftkings", "prop_type": "points",
                                  "line_value": str(line), "opened_at": day + "T15:00:00Z",
                                  "market": {"type": "over_under", "over_odds": -110, "under_odds": -110}})
    gp, _, _ = quarantine(props, games, "bdl_opening_player_props")
    go, _, _ = quarantine(odds, games, "bdl_opening_game_odds")
    return gp, go, stats, games, props


def test_h1_detects_unpriced_blowout_minutes_and_mechanism():
    gp, go, stats, games, _ = world(1, book_knows_blowouts=False)
    res = run(gp, go, stats, games)
    assert res["checks"]["mechanism_minutes_lower_in_blowouts"]
    assert res["primary_blowout_unders"]["power"]["excess"] > 0.1
    assert res["verdict"] == "PASS"


def test_h1_fails_when_book_prices_blowouts():
    gp, go, stats, games, _ = world(2, book_knows_blowouts=True)
    res = run(gp, go, stats, games)
    assert res["verdict"] == "FAIL"


def test_h1_refuses_ungated_inputs():
    _, go, stats, games, raw_props = world(3)
    with pytest.raises(UnverifiedOddsError):
        run(raw_props, go, stats, games)
