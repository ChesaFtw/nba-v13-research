import random
from datetime import date, timedelta

import scripts.log_live_odds as logger
from scripts.cross_market_consistency import build, run


def consistency_world(seed=0, games=500, stale=1.5):
    rng = random.Random(seed)
    teams = {t: [rng.uniform(6, 26) for _ in range(5)] for t in range(1, 9)}
    G, ODDS, PROPS, STATS = [], [], [], []
    for i in range(games):
        day = (date(2025, 10, 22) + timedelta(days=i // 3)).isoformat()
        h, a = rng.sample(sorted(teams), 2)
        gid = i + 1
        pts = {}
        for team in (h, a):
            shade = rng.choice((-stale, 0.0, stale))  # the props desk is stale for this team-game
            total = 0
            for k, mu in enumerate(teams[team]):
                pid = team * 100 + k
                x = max(0, round(rng.gauss(mu, 5)))
                total += x
                STATS.append({"pts": x, "player": {"id": pid}, "team": {"id": team}, "game": {"id": gid, "date": day}})
                PROPS.append({"id": len(PROPS), "game_id": gid, "player_id": pid, "vendor": "caesars", "prop_type": "points",
                              "line_value": str(round(mu + shade) + 0.5), "opened_at": day + "T15:00:00Z",
                              "market": {"type": "over_under", "over_odds": -110, "under_odds": -110}})
            pts[team] = total
        exp_h, exp_a = sum(teams[h]), sum(teams[a])
        G.append({"id": gid, "date": day, "datetime": day + "T23:00:00Z", "home_team": {"id": h}, "visitor_team": {"id": a},
                  "home_team_score": pts[h], "visitor_team_score": pts[a], "status": "Final"})
        ODDS.append({"id": gid, "game_id": gid, "vendor": "draftkings", "opened_at": day + "T14:00:00Z",
                     "spread_home_value": str(-(exp_h - exp_a)), "total_value": str(exp_h + exp_a)})
    # warm-up rows so players have a team before their first scored game
    for team, mus in teams.items():
        for k in range(5):
            STATS.append({"pts": 0, "player": {"id": team * 100 + k}, "team": {"id": team}, "game": {"id": -team, "date": "2025-10-01"}})
    return G, ODDS, PROPS, STATS


def test_consistency_detects_stale_props():
    rows = build(*consistency_world())
    res = run(rows, min_train=100)
    assert res["team_games_scored"] > 100
    assert res["corr_gap_with_player_points_minus_lines"] > 0.2
    assert res["rule_bets"] > 50 and res["rule_roi_flat_pct"] > 0


def test_consistency_no_edge_when_props_are_right():
    res = run(build(*consistency_world(seed=1, stale=0.0)), min_train=100)
    assert res["rule_roi_flat_pct"] is None or res["rule_roi_flat_pct"] < 8


def test_live_logger_stamps_rows():
    from datetime import datetime, timezone
    from scripts.acceptance_round9 import _Session
    obs, raw, manifest = logger.snapshot("k", now=datetime(2026, 10, 28, 18, tzinfo=timezone.utc), session=_Session(), commit="c")
    assert manifest["upcoming_games"] == 1 and len(obs) == 6
    assert all(o["captured_at_utc"] and o["tip_utc"] == "2026-10-28T23:30:00+00:00" for o in obs)


def test_power_devig_and_prereg_runner():
    from v13.pricing import power_devig_pair, no_vig_pair
    po, pu = power_devig_pair(-178, 140)
    assert abs(po + pu - 1) < 1e-9
    mo, mu = no_vig_pair(-178, 140)
    assert po > mo  # power gives the favourite more fair probability
    from tests.test_pricing_and_v2 import synthetic_season
    from scripts.prereg_threes_unders_2024 import run as prereg
    feats, lines, games = synthetic_season(6, priced=True)
    # one line opened after tip must be excluded, never graded
    lines[-1]["opened_at"] = "2099-01-01T00:00:00Z"
    out = prereg(feats, lines, games)
    assert out["verdict"] in ("PASS", "FAIL", "UNTESTABLE")
    assert set(out["checks"]) == {"unders_n_ge_150", "roi_gt_0", "excess_ge_2pts_multiplicative",
                                  "excess_ge_2pts_power", "z_ge_2_multiplicative", "z_ge_2_power"}
    assert "power" in out["primary_unders"] or out["primary_unders"]["decided"] == 0
    assert out["verdict"] != "PASS"  # fairly priced synthetic market has no edge


def test_selection_calibration_reports():
    from scripts.rank_and_stake import selection_calibration
    rows = [{"outcome": "win" if i % 2 else "loss", "price": -110, "p": 0.56, "ev": 0.07, "p_market": 0.5}
            for i in range(100)]
    r = selection_calibration(rows)
    assert r["realized_hit"] == 0.5 and abs(r["realized_edge_share"]) < 1e-9
