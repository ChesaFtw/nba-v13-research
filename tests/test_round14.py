import math
import random
from datetime import date, timedelta

import numpy as np
import pytest

from v13.provenance import UnverifiedOddsError, quarantine
from v13.r14_models import PoissonGLM, nb_logpmf, nb_pmf_vector, threes_logpmf, threes_over_prob

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


def league(seed=0, market_efficient=False, days=150, height_effect=0.06):
    """6 teams x 9 players. Rebounds = minutes x opponent misses x share x exp(height effect);
    3PM = Binomial(Poisson attempts, true %). A naive market prices from season-to-date averages;
    an efficient market prices from the true means."""
    rng = random.Random(seed)
    teams = {}
    for t in range(1, 7):
        roster = []
        for k in range(9):
            pos = "C" if k < 2 else ("F" if k < 5 else "G")
            h = {"C": 84, "F": 80, "G": 76}[pos] + rng.randint(-3, 3) + (t - 3)
            roster.append({"id": t * 100 + k, "pos": pos, "h": h, "min": rng.uniform(14, 34),
                           "a3": rng.uniform(0.02, 0.25) if pos != "C" else 0.01, "p3": rng.uniform(0.30, 0.42),
                           "sd": {"C": 0.30, "F": 0.18, "G": 0.10}[pos], "so": {"C": 0.12, "F": 0.06, "G": 0.03}[pos]})
        teams[t] = {"roster": roster, "fg": rng.uniform(0.40, 0.52), "fga": rng.uniform(78, 96)}
    games, stats, props, raw2024 = [], [], [], []
    run_avg = {}
    gid = 0
    for season, start in STARTS.items():
        run_avg = {}
        for d in range(days):
            day = (start + timedelta(days=d)).isoformat()
            order = list(teams); rng.shuffle(order)
            for i in (0, 2):
                h_, a_ = order[i], order[i + 1]; gid += 1
                games.append({"id": gid, "date": day, "datetime": day + "T23:30:00Z", "season": season, "period": 4,
                              "home_team": {"id": h_}, "visitor_team": {"id": a_}, "status": "Final", "postseason": False})
                lines = {}
                for team, opp in ((h_, a_), (a_, h_)):
                    T = teams[team]; tot = sum(p["min"] for p in T["roster"])
                    lines[team] = []
                    for p in T["roster"]:
                        m = max(2.0, 240 * p["min"] / tot + rng.gauss(0, 2.5))
                        fga = poisson(rng, T["fga"] * m / 240)
                        fgm = sum(rng.random() < T["fg"] for _ in range(fga))
                        att = poisson(rng, p["a3"] * m)
                        made = sum(rng.random() < p["p3"] for _ in range(att))
                        lines[team].append([p, m, fga, fgm, att, made])
                miss = {t: sum(x[2] - x[3] for x in lines[t]) for t in lines}
                size = {t: sum(x[0]["h"] * x[1] for x in lines[t]) / sum(x[1] for x in lines[t]) for t in lines}
                for team, opp in ((h_, a_), (a_, h_)):
                    for p, m, fga, fgm, att, made in lines[team]:
                        hd = p["h"] - size[opp]
                        mu_d = m / 48 * miss[opp] * p["sd"] * math.exp(height_effect * hd)
                        mu_o = m / 48 * miss[team] * p["so"]
                        dreb, oreb = poisson(rng, mu_d), poisson(rng, mu_o)
                        stats.append({"min": f"{int(m)}:{int((m % 1) * 60):02d}", "pts": 2 * fgm + made, "reb": dreb + oreb,
                                      "dreb": dreb, "oreb": oreb, "ast": 2, "fga": fga, "fgm": fgm, "fg3a": att, "fg3m": made,
                                      "pf": 2, "player": {"id": p["id"], "height": f"{p['h'] // 12}-{p['h'] % 12}",
                                                          "position": p["pos"], "draft_year": 2018},
                                      "team": {"id": team}, "game": {"id": gid, "date": day, "season": season}})
                        ra = run_avg.setdefault(p["id"], {"n": 0, "reb": 0.0, "3pm": 0.0})
                        if season in (2024, 2025) and ra["n"] >= 5:
                            exp_m = 240 * p["min"] / sum(q["min"] for q in teams[team]["roster"])
                            if market_efficient:
                                mu_r = exp_m / 48 * (teams[opp]["fga"] * (1 - teams[opp]["fg"])) * p["sd"] * math.exp(height_effect * hd) \
                                    + exp_m / 48 * (teams[team]["fga"] * (1 - teams[team]["fg"])) * p["so"]
                                mu_3 = p["a3"] * exp_m * p["p3"]
                            else:
                                mu_r, mu_3 = ra["reb"] / ra["n"], ra["3pm"] / ra["n"]
                            for ptype, mu, actual_line in (("rebounds", mu_r, None), ("threes", mu_3, None)):
                                if mu < 0.3:
                                    continue
                                line = math.floor(mu) + 0.5
                                po = pois_over(mu, line)
                                row = {"game_id": gid, "player_id": p["id"], "vendor": "draftkings", "prop_type": ptype,
                                       "line_value": str(line), "opened_at": day + "T15:00:00Z",
                                       "market": {"type": "over_under", "over_odds": to_am(po), "under_odds": to_am(1 - po)}}
                                (props if season == 2025 else raw2024).append(row)
                        ra["n"] += 1; ra["reb"] += dreb + oreb; ra["3pm"] += made
    gated, _, _ = quarantine(props, games, "bdl_opening_player_props")
    return stats, games, gated, props, raw2024


@pytest.fixture(scope="module")
def naive():
    return league(seed=3)


def test_threes_compound_probability_matches_brute_force():
    mu, phi, pct, line = 5.0, 0.15, 0.37, 1.5
    pa = nb_pmf_vector(mu, phi, 60)
    brute = 0.0
    for a in range(61):
        brute += pa[a] * sum(math.comb(a, m) * pct ** m * (1 - pct) ** (a - m) for m in range(2, a + 1))
    assert threes_over_prob(mu, phi, pct, line, amax=60) == pytest.approx(brute, abs=1e-6)
    assert sum(math.exp(threes_logpmf(m, mu, phi, pct, amax=60)) for m in range(0, 30)) == pytest.approx(1.0, abs=1e-6)
    assert sum(math.exp(nb_logpmf(k, 7.0, 0.2)) for k in range(200)) == pytest.approx(1.0, abs=1e-6)


def test_poisson_glm_recovers_effect_with_offset():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 2)); off = rng.uniform(1, 6, 5000)
    y = rng.poisson(off * np.exp(0.3 * x[:, 0] - 0.2 * x[:, 1]))
    g = PoissonGLM(ridge=0.0).fit(x, y, off)
    per_sd = g.beta_[1:] / g.sd_
    assert per_sd[0] == pytest.approx(0.3, abs=0.04) and per_sd[1] == pytest.approx(-0.2, abs=0.04)
    assert g.beta_[0] == pytest.approx(0.0, abs=0.05)


def test_r14_features_use_only_earlier_days(naive):
    from v13.r14_features import build
    stats, games, *_ = naive
    last = max(str(g["date"]) for g in games)
    before = build(stats, games)
    changed = [dict(r, reb=30, dreb=25, oreb=5, fg3a=15, fg3m=15, fga=40, fgm=1) if r["game"]["date"] == last else r for r in stats]
    after = build(changed, games)
    keys = [k for k in before if any(g["date"] == last and str(g["id"]) == k[0] for g in games)]
    assert keys
    for k in keys:
        assert before[k] == after[k]


def test_incremental_test_rejects_redundant_feature():
    from scripts.r14_research import evaluate_incremental
    rng = np.random.default_rng(1)
    rows = []
    for i in range(4000):
        z = rng.normal()
        p = 1 / (1 + math.exp(-(0.8 * z)))
        rows.append({"game_id": str(i // 4), "date": "2026-01-01" if i < 2400 else "2026-03-01",
                     "y": int(rng.random() < p), "lp": 0.0, "f": {"a": z, "copy": z + rng.normal() * 1e-3, "noise": rng.normal()}})
    alone = evaluate_incremental(rows, "a")
    assert alone["delta_logloss"] < 0 and alone["delta_95ci"][1] < 0
    redundant = evaluate_incremental(rows, "copy", ("a",))
    assert redundant["delta_95ci"][1] >= 0 or abs(redundant["delta_logloss"]) < 1e-3


def test_full_run_finds_structure_when_market_is_naive(naive):
    from scripts.r14_research import run
    stats, games, gated, _, raw2024 = naive
    res = run(stats, games, gated, raw2024)
    pa = res["part_a"]["fold_2025"]
    assert pa["rebounds_full_vs_b0"]["mean_logscore_gain"] > 0
    assert pa["rebound_glm_coefficients_per_sd"]["opp_size"]["coef"] < 0  # bigger opponents, fewer boards
    assert res["part_a"]["fold_2024"]["threes_v1_vs_v0"]["n"] > 100
    b = res["part_b"]["results"]
    assert b["rebounds|edge_struct"]["status"] == "tested"
    assert b["rebounds|edge_struct"]["delta_logloss"] < 0
    assert res["r13_threes_diagnosis"]["reproduction"]["status"] == "tested"
    assert "edge_deciles" in res["r13_threes_diagnosis"]
    assert res["replication_2024_25"]["evidence_class"] == "exploratory_unverified_prices"
    assert res["tracking_gate"]["passed"] is False


def test_no_information_when_market_is_efficient():
    from scripts.r14_research import run
    stats, games, gated, _, _ = league(seed=3, market_efficient=True)
    b = run(stats, games, gated)["part_b"]["results"]
    assert b["rebounds|edge_struct"].get("verdict") != "PASS"
    assert b["threes|edge_v1"].get("verdict") != "PASS"


def test_refuses_ungated(naive):
    from scripts.r14_research import run
    stats, games, _, raw, _ = naive
    with pytest.raises(UnverifiedOddsError):
        run(stats, games, raw)


def test_vacated_audit_counts_trade_phantoms():
    from scripts.r14_research import vacated_audit
    games, stats = [], []
    for i in range(1, 9):
        games.append({"id": i, "date": f"2025-11-{i:02d}", "season": 2025, "postseason": False,
                      "home_team": {"id": 1}, "visitor_team": {"id": 2 if i < 5 else 3}})
    for i in range(1, 9):
        for pid in (10, 11):
            team = 1
            if pid == 11 and i >= 5:
                team = 3  # traded to team 3 after game 4, plays for team 3 from game 6
                if i == 5:
                    continue
            stats.append({"min": "30:00", "player": {"id": pid}, "team": {"id": team}, "game": {"id": i, "date": f"2025-11-{i:02d}"}})
    out = vacated_audit(stats, games, 2025)
    assert out["flagged_absences"] >= 1 and out["next_appearance_other_team_(trade)"] >= 1


def test_tracking_audit_and_gate():
    from scripts.collect_tracking import audit
    from scripts.r14_research import tracking_gate
    rows = []
    for gid in (1, 2):
        for per, val in ((0, 10), (1, 3), (2, 3), (3, 2), (4, 2 if gid == 1 else 9)):
            rows.append({"game": {"id": gid, "date": "2025-11-01"}, "player": {"id": 7}, "period": per,
                         "rebound_chances_total": val, "touches": None})
    full, aud = audit(rows)
    assert len(full) == 2 and aud["period_consistency"]["rebound_chances_total"] == 0.5
    assert aud["field_nonnull_period0"]["touches"] == 0.0 and aud["duplicate_keys"] == 0
    games = [{"id": g, "season": s} for g, s in ((1, 2023), (2, 2024), (3, 2025))]
    stats = [{"min": "20:00", "game": {"id": g}, "player": {"id": 7}} for g in (1, 2, 3)]
    tr = {(str(g), "7"): {"rebound_chances_total": 5} for g in (1, 2, 3)}
    assert tracking_gate(tr, stats, games, [])["passed"] is True
    tr.pop(("2", "7"))
    assert tracking_gate(tr, stats, games, [])["passed"] is False


def test_frozen_shadow_recipe_is_emitted(naive):
    from scripts.r14_research import run
    stats, games, gated, _, _ = naive
    fr = run(stats, games, gated)["frozen_threes_shadow_recipe"]
    assert fr["status"] == "frozen" and fr["trained_on_seasons"] == [2023, 2024, 2025]
    assert set(fr["recalibration_logit"]) == {"intercept", "market_logit", "edge_z"}
