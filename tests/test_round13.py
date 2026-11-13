import math
import random
from datetime import date, timedelta

import pytest

from v13.minutes_model import FEATURES, MinutesModel, build, ewma, nb_dispersion, nb_over_prob
from v13.provenance import UnverifiedOddsError, quarantine


def to_am(p):
    p = min(max(p, 0.03), 0.97)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def poisson_over(mu, line):
    k = math.floor(line)
    return 1 - sum(math.exp(-mu) * mu ** j / math.factorial(j) for j in range(k + 1))


SEASON_STARTS = {2023: date(2023, 10, 24), 2024: date(2024, 10, 22), 2025: date(2025, 10, 22)}


def league(seed=0, market_knows_absences=False, days=160):
    """6 teams x 9 players. Players miss stretches of games; their minutes go to present teammates
    in proportion to base minutes. Points ~ Poisson(rate x minutes). 2025-26 props are priced from
    full-roster minutes (market ignores absences) unless market_knows_absences."""
    rng = random.Random(seed)
    teams = {t: [(t * 100 + k, rng.uniform(8, 36), rng.uniform(0.3, 0.75)) for k in range(9)] for t in range(1, 7)}
    out_until = {}
    stats, games, props = [], [], []
    gid = 0
    for season, start in SEASON_STARTS.items():
        team_games = {t: 0 for t in teams}
        out_until = {}
        for d in range(days):
            day = (start + timedelta(days=d)).isoformat()
            order = list(teams); rng.shuffle(order)
            for i in (0, 2):
                h, a = order[i], order[i + 1]; gid += 1
                games.append({"id": gid, "date": day, "datetime": day + "T23:30:00Z", "season": season, "period": 4,
                              "home_team": {"id": h}, "visitor_team": {"id": a}, "status": "Final", "postseason": False})
                for team in (h, a):
                    team_games[team] += 1
                    n = team_games[team]
                    roster = teams[team]
                    for pid, base, _ in roster:
                        if out_until.get(pid, 0) < n and rng.random() < 0.04:
                            out_until[pid] = n + rng.randint(2, 6)
                    present = [(pid, base, rate) for pid, base, rate in roster if out_until.get(pid, 0) < n]
                    tot = sum(b for _, b, _ in present); full = sum(b for _, b, _ in roster)
                    for pid, base, rate in present:
                        exp_min = min(42.0, 240.0 * base / tot)
                        mins = max(1.0, exp_min + rng.gauss(0, 2.5))
                        pts = sum(rng.random() < rate * mins / 60 for _ in range(60))
                        stats.append({"min": f"{int(mins)}:{int((mins % 1) * 60):02d}", "pts": pts, "reb": 3, "ast": 2,
                                      "fg3m": 1, "pf": 2, "player": {"id": pid}, "team": {"id": team},
                                      "game": {"id": gid, "date": day, "season": season}})
                        if season == 2025:
                            mkt_min = exp_min if market_knows_absences else min(42.0, 240.0 * base / full)
                            mu = rate * mkt_min
                            line = math.floor(mu) + 0.5
                            po = poisson_over(mu, line)
                            props.append({"game_id": gid, "player_id": pid, "vendor": "draftkings", "prop_type": "points",
                                          "line_value": str(line), "opened_at": day + "T15:00:00Z",
                                          "market": {"type": "over_under", "over_odds": to_am(po), "under_odds": to_am(1 - po)}})
    gated, _, _ = quarantine(props, games, "bdl_opening_player_props")
    return stats, games, gated, props


@pytest.fixture(scope="module")
def blind_world():
    return league(seed=1)


def test_features_use_only_earlier_days(blind_world):
    stats, games, _, _ = blind_world
    last_day = max(str(g["date"]) for g in games)
    before = build(stats, games)
    changed = [dict(r, min="40:00", pts=60) if r["game"]["date"] == last_day else r for r in stats]
    after = build(changed, games)
    keys = [k for k, v in before.items() if v["date"] == last_day]
    assert keys
    for k in keys:
        assert before[k]["features"] == after[k]["features"]
        assert before[k]["rates"] == after[k]["rates"]


def test_vacated_minutes_detected_from_previous_game_only():
    games = [{"id": i, "date": f"2025-11-0{i}", "season": 2025, "period": 4, "postseason": False,
              "home_team": {"id": 1}, "visitor_team": {"id": 2}} for i in range(1, 6)]
    stats = []
    for g in range(1, 6):
        for pid in (10, 11):
            if pid == 11 and g == 4:
                continue  # star misses game 4
            stats.append({"min": "30:00", "pts": 10, "pf": 1, "player": {"id": pid}, "team": {"id": 1},
                          "game": {"id": g, "date": f"2025-11-0{g}"}})
    f = build(stats, games)
    assert f[("4", "10")]["features"]["vacated_min"] == 0          # absence in game 4 not known before game 4
    assert f[("5", "10")]["features"]["vacated_min"] == pytest.approx(30.0)  # known before game 5
    assert f[("5", "11")]["features"]["missed_before"] == 1


def test_overtime_minutes_are_scaled_in_history():
    games = [{"id": i, "date": f"2025-11-0{i}", "season": 2025, "period": 5 if i == 1 else 4, "postseason": False,
              "home_team": {"id": 1}, "visitor_team": {"id": 2}} for i in range(1, 5)]
    stats = [{"min": "53:00" if g == 1 else "48:00", "pts": 10, "player": {"id": 7}, "team": {"id": 1},
              "game": {"id": g, "date": f"2025-11-0{g}"}} for g in range(1, 5)]
    f = build(stats, games)[("4", "7")]["features"]
    assert f["m_l3"] == pytest.approx((53 * 48 / 53 + 48 + 48) / 3)


def test_negative_binomial_probability():
    # tiny dispersion -> Poisson
    assert nb_over_prob(10.0, 1e-6, 9.5) == pytest.approx(poisson_over(10.0, 9.5), abs=1e-3)
    # more dispersion widens tails: P(X > 15.5) rises
    assert nb_over_prob(10.0, 0.3, 15.5) > nb_over_prob(10.0, 0.01, 15.5)
    assert nb_dispersion([5.0] * 4, [5.0, 5.0, 5.0, 5.0]) == pytest.approx(1e-6)
    assert ewma([1.0, 1.0, 1.0], 5) == 1.0


def test_minutes_model_beats_ewma_when_absences_drive_minutes(blind_world):
    from scripts.r13_opportunity import run
    stats, games, gated, _ = blind_world
    res = run(stats, games, gated)
    a = res["part_a_minutes"]
    assert a["checks"]["mae_improvement_ci_upper_lt_0"], a
    assert a["mae"]["model"] < a["mae"]["ewma"]
    assert 0.70 <= a["interval_80_coverage"] <= 0.90
    assert a["coefficients"]["vacated_min"] > 0
    b = res["part_b_information"]["results"]
    assert b["points|model_edge"]["verdict"] == "PASS", b["points|model_edge"]
    assert b["points|vacated_min"]["delta_logloss"] < 0


def test_no_information_when_market_already_knows():
    from scripts.r13_opportunity import run
    stats, games, gated, _ = league(seed=1, market_knows_absences=True)
    b = run(stats, games, gated)["part_b_information"]["results"]
    assert b["points|model_edge"]["verdict"] == "FAIL"
    assert b["points|vacated_min"]["verdict"] == "FAIL"


def test_refuses_ungated_props(blind_world):
    from scripts.r13_opportunity import run
    stats, games, _, raw = blind_world
    with pytest.raises(UnverifiedOddsError):
        run(stats, games, raw)


def test_feature_list_is_complete(blind_world):
    stats, games, _, _ = blind_world
    one = next(iter(build(stats[:3000], games).values()))
    assert set(one["features"]) == set(FEATURES)


def test_inventory_helpers():
    from scripts.r13_data_inventory import field_coverage, prop_inventory, summarize
    rows = [{"period": 0, "touches": 50, "rebound_chances_total": None, "player": {"id": 1}},
            {"period": 0, "touches": None, "rebound_chances_total": None, "player": {"id": 2}},
            {"period": 1, "touches": 9, "rebound_chances_total": 3, "player": {"id": 1}}]
    cov = field_coverage(rows[:2])
    assert cov["touches"] == 0.5 and cov["rebound_chances_total"] == 0.0 and "player" not in cov
    sm = summarize({"status": 200, "body": {"data": rows}}, period_zero=True)
    assert sm["rows"] == 3 and sm["rows_used"] == 2 and sm["key_field_coverage"]["touches"] == 0.5
    assert summarize({"error": "x"}) == {"error": "x"}
    inv = prop_inventory([
        {"prop_type": "points", "vendor": "fanduel", "game_id": 1, "market": {"type": "over_under", "over_odds": -110, "under_odds": -110}},
        {"prop_type": "steals", "vendor": "fanduel", "game_id": 1, "market": {"type": "milestone", "odds": 300}},
    ])
    assert inv["prop_types"]["points"]["two_sided"] == 1 and inv["prop_types"]["steals"].get("two_sided", 0) == 0
    assert inv["games"] == 1


def test_injury_capture_normalize_and_status_changes():
    from scripts.capture_injuries import normalize, status_changes
    raw = {"player": {"id": 5, "first_name": "A", "last_name": "B", "team_id": 9}, "status": "Out",
           "return_date": "Nov 3", "description": "ankle"}
    n1 = normalize(raw, "2026-10-21T15:00:00+00:00")
    n2 = normalize(raw, "2026-10-21T15:30:00+00:00")
    assert n1["player_id"] == 5 and n1["team_id"] == 9 and n1["row_sha256"] == n2["row_sha256"]
    snaps = [("2026-10-21T16:00:00+00:00", [{"player_id": 5, "status": "Out"}, {"player_id": 6, "status": "Questionable"}]),
             ("2026-10-21T15:00:00+00:00", [{"player_id": 6, "status": "Questionable"}]),
             ("2026-10-21T17:00:00+00:00", [{"player_id": 5, "status": "Out"}])]
    ev = status_changes(snaps)
    assert {(e["player_id"], e["old_status"], e["new_status"], e["first_seen_utc"]) for e in ev} == {
        (5, "not_listed", "Out", "2026-10-21T16:00:00+00:00"),
        (6, "Questionable", "not_listed", "2026-10-21T17:00:00+00:00")}
