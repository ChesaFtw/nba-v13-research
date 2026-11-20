import math
import random
from datetime import date, timedelta

import pytest

from v13.matchup_features import build, parse_height
from v13.provenance import UnverifiedOddsError, quarantine


def test_parse_height():
    assert parse_height("6-11") == 83 and parse_height(None) is None and parse_height("x") is None


def league(seed=0, market_knows_size=False, effect=0.25, days=170):
    """8 teams x 8 players; rebounds rise with height advantage over the opponent's size."""
    rng = random.Random(seed)
    teams = {t: [(t * 100 + k, rng.randint(74, 86), rng.uniform(0.15, 0.35)) for k in range(8)] for t in range(1, 9)}
    size = {t: sum(h for _, h, _ in ps) / len(ps) for t, ps in teams.items()}
    stats, games, props = [], [], []
    gid = 0
    for d in range(days):
        day = (date(2025, 10, 22) + timedelta(days=d)).isoformat()
        order = list(teams); rng.shuffle(order)
        for i in range(0, 8, 2):
            h, a = order[i], order[i + 1]; gid += 1
            games.append({"id": gid, "date": day, "datetime": day + "T23:30:00Z", "season": 2025,
                          "home_team": {"id": h}, "visitor_team": {"id": a}, "status": "Final", "postseason": False})
            for team, opp in ((h, a), (a, h)):
                for pid, hgt, rate in teams[team]:
                    mins = 28.0
                    mu = max(0.3, rate * mins + effect * (hgt - size[opp]))
                    reb = sum(rng.random() < mu / 60 for _ in range(60))
                    stats.append({"min": "28:00", "pts": 10, "reb": reb, "ast": 2, "fga": 10, "fgm": 4, "fg3a": 3,
                                  "player": {"id": pid, "height": f"{hgt // 12}-{hgt % 12}", "draft_year": 2018},
                                  "team": {"id": team}, "game": {"id": gid, "date": day}})
                    fair_mu = rate * mins + (effect * (hgt - size[opp]) if market_knows_size else 0.0)
                    line = math.floor(fair_mu) + 0.5
                    # price the over from a Poisson(fair_mu) so the market is right about what it knows
                    k = math.floor(line); cdf = sum(math.exp(-fair_mu) * fair_mu ** j / math.factorial(j) for j in range(k + 1))
                    po = min(max(1 - cdf, 0.05), 0.95)
                    to_am = lambda p: int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))
                    props.append({"game_id": gid, "player_id": pid, "vendor": "draftkings", "prop_type": "rebounds",
                                  "line_value": str(line), "opened_at": day + "T15:00:00Z",
                                  "market": {"type": "over_under", "over_odds": to_am(po), "under_odds": to_am(1 - po)}})
    gated, _, _ = quarantine(props, games, "bdl_opening_player_props")
    return gated, stats, games, props


def test_features_use_only_earlier_games():
    _, stats, games, _ = league(1, days=30)
    f1 = build(stats, games)
    later = [dict(r) for r in stats]
    last_day = max(r["game"]["date"] for r in later)
    for r in later:
        if r["game"]["date"] == last_day:
            r["reb"] = 99
    f2 = build(later, games)
    for k, v in f1.items():
        assert f2[k] == v  # changing the final day's results changes no feature


def test_height_signal_found_when_price_ignores_it_and_not_when_priced():
    from scripts.feature_info_test import run
    gated, stats, games, _ = league(2, market_knows_size=False)
    res = run(gated, stats, games)
    assert res["results"]["rebounds|height_diff"]["verdict"] == "PASS"
    gated2, stats2, games2, _ = league(3, market_knows_size=True)
    res2 = run(gated2, stats2, games2)
    assert res2["results"]["rebounds|height_diff"]["verdict"] == "FAIL"


def test_info_test_refuses_ungated_props():
    from scripts.feature_info_test import run
    _, stats, games, raw = league(4, days=20)
    with pytest.raises(UnverifiedOddsError):
        run(raw, stats, games)


def test_bh_controls_false_discoveries():
    from scripts.feature_info_test import benjamini_hochberg
    assert benjamini_hochberg([0.001, 0.2, 0.5, 0.04]) == [True, False, False, True]
    assert benjamini_hochberg([0.03, 0.2, 0.5, 0.6]) == [False, False, False, False]
