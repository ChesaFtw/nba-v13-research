import random
from datetime import date, timedelta

import pytest

from scripts.book_disagreement import FROZEN, opening_quotes, pick_bets, run
from v13.provenance import UnverifiedOddsError, quarantine


def to_am(p):
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def quote(p_over, vig=0.045):
    """Two-sided American prices with margin split evenly."""
    return to_am(min(0.95, p_over + vig / 2)), to_am(min(0.95, 1 - p_over + vig / 2))


def world(seed=0, soft_book_wrong=True, n_days=150, stagger_min=0):
    """4 books quote rebounds overs. Three sharp books quote close to the truth. The soft book is
    off by 0.10 on 30% of props. If soft_book_wrong, the sharp books are right (the truth is their
    probability); otherwise the soft book is right and the sharp books are the ones who are off."""
    rng = random.Random(seed)
    games, props, stats = [], [], []
    gid = 0
    for d in range(n_days):
        day = (date(2025, 10, 22) + timedelta(days=d)).isoformat()
        for _ in range(4):
            gid += 1
            games.append({"id": gid, "date": day, "datetime": day + "T23:30:00Z", "season": 2025,
                          "home_team": {"id": 1}, "visitor_team": {"id": 2}, "status": "Final", "postseason": False})
            for pid in range(gid * 10, gid * 10 + 5):
                sharp = rng.uniform(0.35, 0.65)
                soft = sharp + (rng.choice((-0.10, 0.10)) if rng.random() < 0.30 else 0.0)
                soft = min(max(soft, 0.1), 0.9)
                truth = sharp if soft_book_wrong else soft
                line = 7.5
                for k, v in enumerate(("draftkings", "fanduel", "betmgm", "caesars")):
                    p = soft if v == "caesars" else sharp + rng.uniform(-0.005, 0.005)
                    o, u = quote(p)
                    minute = 0 if v != "caesars" else stagger_min
                    props.append({"game_id": gid, "player_id": pid, "vendor": v, "prop_type": "rebounds",
                                  "line_value": str(line), "opened_at": f"{day}T15:{minute:02d}:00Z" if minute < 60
                                  else f"{day}T{15 + minute // 60}:{minute % 60:02d}:00Z",
                                  "market": {"type": "over_under", "over_odds": o, "under_odds": u}})
                reb = 9 if rng.random() < truth else 6
                stats.append({"reb": reb, "pts": 10, "ast": 2, "fg3m": 1, "player": {"id": pid}, "game": {"id": gid}})
    gated, _, _ = quarantine(props, games, "bdl_opening_player_props")
    return gated, stats, props


def test_price_outlier_passes_when_soft_book_is_wrong():
    gated, stats, _ = world(seed=3, soft_book_wrong=True)
    res = run(gated, stats)
    h = res["hypotheses"]["H12a_price_outlier"]
    assert h["verdict"] == "PASS", h["checks"]
    assert h["primary"]["roi_pct"] > 5
    assert all(r["vendor"] == "caesars" for r in h["sample"])
    # betting against the outlier should match the consensus, not beat it
    assert abs(h["primary"]["vs_consensus"]["z"]) < 2.5


def test_price_outlier_fails_when_soft_book_is_right():
    gated, stats, _ = world(seed=3, soft_book_wrong=False)
    h = run(gated, stats)["hypotheses"]["H12a_price_outlier"]
    assert h["verdict"] == "FAIL"
    assert h["primary"]["roi_pct"] < 0


def test_quotes_outside_the_window_are_not_compared():
    gated, stats, _ = world(stagger_min=FROZEN["window_min"] + 30)
    pb, _, _ = pick_bets(opening_quotes(gated))
    assert all(b["vendor"] != "caesars" for b in pb)


def test_one_bet_per_player_market_and_line_outlier_side():
    t = "2025-11-01T15:00:00Z"
    rows = []
    for v, line in (("draftkings", 8.5), ("fanduel", 8.5), ("betmgm", 8.5), ("caesars", 7.5)):
        rows.append({"game_id": 1, "player_id": 9, "vendor": v, "prop_type": "rebounds", "line_value": str(line),
                     "opened_at": t, "market": {"type": "over_under", "over_odds": -110, "under_odds": -110}})
    groups = opening_quotes(rows)
    pb, lb, _ = pick_bets(groups)
    assert len(lb) == 1 and lb[0]["vendor"] == "caesars" and lb[0]["side"] == "over" and lb[0]["gap"] == 1.0
    assert pb == []  # identical prices at the same line: no price outlier


def test_line_outlier_skips_heavily_juiced_side():
    t = "2025-11-01T15:00:00Z"
    rows = [{"game_id": 1, "player_id": 9, "vendor": v, "prop_type": "rebounds", "line_value": str(line),
             "opened_at": t, "market": {"type": "over_under", "over_odds": oo, "under_odds": uo}}
            for v, line, oo, uo in (("draftkings", 8.5, -110, -110), ("fanduel", 8.5, -110, -110),
                                    ("betmgm", 8.5, -110, -110), ("caesars", 7.5, -175, 135))]
    _, lb, _ = pick_bets(opening_quotes(rows))
    assert lb == []


def test_refuses_ungated_props():
    _, stats, raw = world(n_days=3)
    with pytest.raises(UnverifiedOddsError):
        run(raw, stats)


def test_injury_report_filename_times_and_links():
    from scripts.injury_report_probe import candidate_urls, extract_links, report_time, spread_sample
    hourly = report_time("https://ak-static.cms.nba.com/referee/injury/Injury-Report_2025-01-15_05PM.pdf")
    assert hourly["eastern"].startswith("2025-01-15T17:00") and hourly["utc"].startswith("2025-01-15T22:00")
    assert hourly["format"] == "hourly"
    quarter = report_time("Injury-Report_2025-12-22_01_30PM.pdf")
    assert quarter["eastern"].startswith("2025-12-22T13:30") and quarter["format"] == "quarter_hour"
    assert report_time("Injury-Report_2025-12-22_12AM.pdf")["eastern"].startswith("2025-12-22T00:00")
    assert report_time("other.pdf") is None
    html = ('<a href="https://ak-static.cms.nba.com/referee/injury/Injury-Report_2025-11-05_05PM.pdf">x</a>'
            '<a href="https://ak-static.cms.nba.com/referee/injury/Injury-Report_2025-11-05_05PM.pdf">dup</a>')
    assert len(extract_links(html)) == 1
    assert len(candidate_urls("2025-11-05")) == 24 * 5
    assert spread_sample(list(range(100)), 5) == [0, 25, 50, 74, 99] and spread_sample([1, 2], 5) == [1, 2]
