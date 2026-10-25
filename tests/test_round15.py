import json
import math
import random
from datetime import date, timedelta

import numpy as np
import pytest

from v13.provenance import UnverifiedOddsError, quarantine

STARTS = {2023: date(2023, 10, 24), 2024: date(2024, 10, 22), 2025: date(2025, 10, 22)}


def to_am(p):
    p = min(max(p, 0.03), 0.97)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def pois_over(mu, line):
    k = math.floor(line)
    return 1 - sum(math.exp(-mu) * mu ** j / math.factorial(j) for j in range(k + 1))


def poisson(rng, lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def league(seed=0, efficient=False, days=150):
    """Assists = Poisson(minutes x pass rate x conversion); the pass rate drifts within the season, so
    passes (a high-count opportunity measure) track it better than recent assists. Steals via deflections,
    blocks via rim contests. Players take single-game rests (back next game) and multi-game injuries."""
    rng = random.Random(seed)
    teams = {t: [{"id": t * 100 + k, "min": rng.uniform(16, 34), "pr": rng.uniform(0.3, 1.6), "app": rng.uniform(0.08, 0.14),
                  "dr": rng.uniform(0.03, 0.12), "spd": rng.uniform(0.25, 0.45), "rr": rng.uniform(0.05, 0.35),
                  "bpc": rng.uniform(0.05, 0.15), "fga": rng.uniform(0.2, 0.5), "reb": rng.uniform(0.1, 0.35)}
                 for k in range(9)] for t in range(1, 7)}
    games, stats, props, tracking = [], [], [], {}
    gid = 0
    for season, start in STARTS.items():
        out_until, n_games, avg = {}, {t: 0 for t in teams}, {}
        for p in [p for r in teams.values() for p in r]:
            p["pr_t"] = p["pr"]
        for d in range(days):
            day = (start + timedelta(days=d)).isoformat()
            order = list(teams); rng.shuffle(order)
            for i in (0, 2):
                h, a = order[i], order[i + 1]; gid += 1
                games.append({"id": gid, "date": day, "datetime": day + "T23:30:00Z", "season": season, "period": 4,
                              "home_team": {"id": h}, "visitor_team": {"id": a}, "status": "Final", "postseason": False})
                for team in (h, a):
                    n_games[team] += 1; n = n_games[team]
                    for p in teams[team]:
                        if out_until.get(p["id"], 0) < n:
                            u = rng.random()
                            if u < 0.05:
                                out_until[p["id"]] = n          # rest: misses this game only
                            elif u < 0.06:
                                out_until[p["id"]] = n + rng.randint(4, 8)  # injury
                    present = [p for p in teams[team] if out_until.get(p["id"], 0) < n]
                    tot = sum(p["min"] for p in present)
                    for p in present:
                        p["pr_t"] = min(2.2, max(0.2, p["pr_t"] + rng.gauss(0, 0.05)))
                        m = max(3.0, min(42.0, 240 * p["min"] / tot) + rng.gauss(0, 2.5))
                        passes = poisson(rng, p["pr_t"] * m)
                        ast = sum(rng.random() < p["app"] for _ in range(passes))
                        defl = poisson(rng, p["dr"] * m); stl = sum(rng.random() < p["spd"] for _ in range(defl))
                        rim = poisson(rng, p["rr"] * m); blk = sum(rng.random() < p["bpc"] for _ in range(rim))
                        fga = poisson(rng, p["fga"] * m); fgm = sum(rng.random() < 0.47 for _ in range(fga))
                        reb = poisson(rng, p["reb"] * m)
                        stats.append({"min": f"{int(m)}:{int((m % 1) * 60):02d}", "pts": 2 * fgm, "reb": reb, "oreb": 1, "dreb": reb,
                                      "ast": ast, "stl": stl, "blk": blk, "turnover": 1, "fga": fga, "fgm": fgm, "fg3a": 0, "fg3m": 0,
                                      "fta": 0, "ftm": 0, "pf": 2, "player": {"id": p["id"]}, "team": {"id": team},
                                      "game": {"id": gid, "date": day, "season": season}})
                        tracking[(str(gid), str(p["id"]))] = {"passes": passes, "touches": passes * 2, "deflections": defl,
                                                              "defended_at_rim_fga": rim, "usage_percentage": 0.2}
                        av = avg.setdefault(p["id"], {"n": 0, "ast": 0.0, "stl": 0.0, "blk": 0.0})
                        if season == 2025 and av["n"] >= 5:
                            exp_m = min(42.0, 240 * p["min"] / tot)
                            for ptype, true_mu, naive in (("assists", exp_m * p["pr_t"] * p["app"], av["ast"] / av["n"]),
                                                          ("steals", exp_m * p["dr"] * p["spd"], av["stl"] / av["n"]),
                                                          ("blocks", exp_m * p["rr"] * p["bpc"], av["blk"] / av["n"])):
                                mu = true_mu if efficient else naive
                                if mu < 0.2:
                                    continue
                                line = math.floor(mu) + 0.5; po = pois_over(mu, line)
                                props.append({"game_id": gid, "player_id": p["id"], "vendor": "draftkings", "prop_type": ptype,
                                              "line_value": str(line), "opened_at": day + "T15:00:00Z",
                                              "market": {"type": "over_under", "over_odds": to_am(po), "under_odds": to_am(1 - po)}})
                        av["n"] += 1; av["ast"] += ast; av["stl"] += stl; av["blk"] += blk
    gated, _, _ = quarantine(props, games, "bdl_opening_player_props")
    return stats, games, gated, props, tracking


@pytest.fixture(scope="module")
def naive():
    return league(seed=5)


def test_r15_features_use_only_earlier_days(naive):
    from v13.r15_features import build
    stats, games, _, _, tracking = naive
    last = max(str(g["date"]) for g in games)
    before = build(stats, games, tracking)
    changed = [dict(r, ast=20, stl=9, blk=9, fga=30, fgm=1, pts=2) if r["game"]["date"] == last else r for r in stats]
    tr2 = {k: (dict(v, passes=99, deflections=40) if any(str(g["id"]) == k[0] and g["date"] == last for g in games[-4:]) else v)
           for k, v in tracking.items()}
    after = build(changed, games, tr2)
    keys = [k for k in before if any(str(g["id"]) == k[0] and g["date"] == last for g in games[-4:])]
    assert keys
    for k in keys:
        assert before[k] == after[k]


def test_absence_rule_prefers_multi_game_absences(naive):
    from scripts.r15_research import absence_reliability, choose_k
    stats, games, *_ = naive
    k1 = absence_reliability(stats, games, 2023, 1)
    k2 = absence_reliability(stats, games, 2023, 2)
    assert k1["share_played_anyway"] > k2["share_played_anyway"]
    k, rel = choose_k(stats, games)
    assert k in (2, 3) and len(rel) == 3


def test_feasibility_verdicts():
    from scripts.r15_research import feasibility
    rows = [{"prop_type": "assists", "vendor": "caesars", "market": {"type": "over_under", "over_odds": -110, "under_odds": -110}}] * 2500
    rows += [{"prop_type": "points_1q", "market": {"type": "over_under", "over_odds": -110, "under_odds": -110}},
             {"prop_type": "double_double", "market": {"type": "milestone", "odds": 150}},
             {"prop_type": "steals", "market": {"type": "over_under", "over_odds": -110, "under_odds": -110}}]
    f = feasibility(rows)["by_prop_type"]
    assert f["assists"]["verdict"] == "modeled in R15" and f["assists"]["two_sided_caesars"] == 2500
    assert "play-by-play" in f["points_1q"]["verdict"] and "milestone" in f["double_double"]["verdict"]
    assert "insufficient" in f["steals"]["verdict"]


def test_full_run_naive_market(naive):
    from scripts.r15_research import run
    stats, games, gated, _, tracking = naive
    res = run(stats, games, gated, None, tracking)
    assert all(res["tracking_field_gate"][f] for f in ("passes", "deflections", "defended_at_rim_fga"))
    pa = res["part_a"]["fold_2025"]
    assert pa["assists_passes_vs_recent_assists_offset"]["mean_logscore_gain"] > 0
    assert pa["steals_opp_vs_base"]["n"] > 100 and pa["blocks_opp_vs_base"]["n"] > 100
    assert "pr_joint_vs_independent" in pa
    b = res["part_b"]["results"]
    assert b["assists|edge_opp"]["status"] == "tested"
    assert set(b["assists|edge_opp"]["by_model_direction"]) == {"leans_over", "leans_under"}
    assert "ece" in b["assists|edge_opp"]["calibration_challenger"]
    assert res["absence_v2"]["chosen_k"] in (2, 3)


def test_no_pass_when_market_efficient():
    from scripts.r15_research import run
    stats, games, gated, _, tracking = league(seed=5, efficient=True)
    b = run(stats, games, gated, None, tracking)["part_b"]["results"]
    for k in ("assists|edge_opp", "steals|edge_opp", "blocks|edge_opp"):
        assert b[k].get("verdict") != "PASS", (k, b[k])


def test_refuses_ungated(naive):
    from scripts.r15_research import run
    stats, games, _, raw, tracking = naive
    with pytest.raises(UnverifiedOddsError):
        run(stats, games, raw, None, tracking)


def _recipe(stats, games):
    from v13.minutes_model import MinutesModel, build
    items = [v for v in build(stats, games).values()]
    mm = MinutesModel().fit(items)
    return {"status": "frozen", "minutes_model_coefficients": mm.coefficients(),
            "minutes_model_standardization": {"mean": list(map(float, mm.mu)), "sd": list(map(float, mm.sd))},
            "phi_threes": 0.2, "edge_standardization": {"mean": 0.0, "sd": 0.5},
            "recalibration_logit": {"intercept": 0.0, "market_logit": 1.0, "edge_z": 0.4}}


def test_ledger_predict_settle_report():
    from scripts.prop_ledger import build_predictions, report, settle, short_list
    rng = random.Random(1)
    games, stats = [], []
    for i in range(1, 13):
        d = f"2026-10-{i + 9:02d}"
        games.append({"id": i, "date": d, "datetime": d + "T23:30:00Z", "season": 2026, "postseason": False,
                      "home_team": {"id": 1}, "visitor_team": {"id": 2}, "status": "Final" if i < 12 else "scheduled"})
        if i < 12:
            for pid, team in ((7, 1), (8, 1), (9, 2)):
                stats.append({"min": "30:00", "pts": 10, "reb": 4, "ast": 3, "fg3m": rng.randint(0, 5), "pf": 2,
                              "player": {"id": pid}, "team": {"id": team}, "game": {"id": i, "date": d}})
    recipe = _recipe(stats, games)
    props = [{"game_id": 12, "player_id": pid, "vendor": "caesars", "prop_type": "threes", "line_value": "0.5",
              "market": {"type": "over_under", "over_odds": 300, "under_odds": -450}} for pid in (7, 8, 9)]
    props.append({"game_id": 12, "player_id": 7, "vendor": "draftkings", "prop_type": "threes", "line_value": "1.5",
                  "market": {"type": "over_under", "over_odds": -110, "under_odds": -110}})
    injuries = [{"player": {"id": 8, "team_id": 1, "first_name": "B", "last_name": "C"}, "status": "Out", "description": "rest"}]
    policy = {"vendor": "caesars", "markets": {"threes": "threes"}, "min_prob_diff": 0.03, "mode": "shadow",
              "bankroll": 10000, "stake_pct": 0.0025}
    rows, skipped = build_predictions(stats, games, props, injuries, recipe, policy, "2026-10-21T18:00:00+00:00", "abc")
    assert len(rows) == 3 and all(r["vendor"] == "caesars" for r in rows)
    r8 = next(r for r in rows if r["player_id"] == "8")
    assert r8["player_injury_status"] == "Out" and r8["qualified"] is False
    assert all(r["stake"] == 0 for r in rows)
    assert any(r["qualified"] for r in rows if r["player_id"] != "8"), rows
    real = dict(policy, mode="real")
    rows_r, _ = build_predictions(stats, games, props, injuries, recipe, real, "2026-10-21T18:00:00+00:00", "abc")
    assert all(r["stake"] == 25.0 for r in rows_r if r["qualified"])
    assert "qualified" in short_list(rows)
    q = next(r for r in rows if r["qualified"])
    closes = [{"prediction_id": q["prediction_id"], "captured_at_utc": "2026-10-21T23:20:00+00:00", "tip_utc": q["tip_utc"],
               "line": q["line"], "over_odds": 200, "under_odds": -260}]
    box = [{"fg3m": 2, "player": {"id": int(q["player_id"])}, "game": {"id": 12}}]
    st = settle([q], closes, box)
    assert st[0]["outcome"] in ("win", "loss") and "clv_ev" in st[0]
    rep = report(rows + [dict(q, captured_at_utc="2026-10-21T22:00:00+00:00", prediction_id="later")], st)
    assert rep["predictions"] == 3 and rep["decided"] == 1
