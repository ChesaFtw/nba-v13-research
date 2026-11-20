import math
import random
from datetime import date, timedelta

import pytest

from v13.team_ratings import RatingModel, game_records, possessions, pregame_projections
from v13.schedule_travel import schedule_travel_features
from v13.ranking import Calibrator, kelly_tier, prepare, select, simulate, walk_forward_score, tau_for_target
from scripts.game_model_v2 import build_rows, run as run_games, normal_sides
from scripts.rank_and_stake import run as rank_run

ABBR = ["BOS", "DEN", "LAL", "MIA", "CHI", "PHX", "NYK", "OKC"]


def synth_league(seed=0, seasons=(2024, 2025), days=150, teams=8, b2b_penalty=0.0):
    """Teams with true offence/defence/pace; box scores consistent with the score.

    b2b_penalty: points per 100 a team loses on the second night of a back-to-back,
    which the synthetic MARKET ignores (a planted, learnable inefficiency)."""
    rng = random.Random(seed)
    off = [rng.gauss(0, 4) for _ in range(teams)]
    dfn = [rng.gauss(0, 4) for _ in range(teams)]
    pace = [rng.gauss(0, 2) for _ in range(teams)]
    games, stats, odds = [], [], []
    gid = 0
    for season in seasons:
        for d in range(days):
            day = (date(season, 10, 22) + timedelta(days=d)).isoformat()
            order = list(range(teams)); rng.shuffle(order)
            if b2b_penalty and d % 2 == 1:
                order = prev_order[:]  # same teams play again tomorrow -> everyone on a back-to-back
                order = order[1::2] + order[0::2]  # swap home/away pairs
            prev_order = order
            for i in range(0, teams, 2):
                h, a = order[i], order[i + 1]
                gid += 1
                exp_poss = 99 + pace[h] + pace[a]
                poss = exp_poss + rng.gauss(0, 3)
                fatigue = b2b_penalty if (b2b_penalty and d % 2 == 1) else 0.0
                hp = round(poss * (114 + 2.5 + off[h] - dfn[a] - fatigue + rng.gauss(0, 11)) / 100)
                ap = round(poss * (114 + off[a] - dfn[h] - fatigue + rng.gauss(0, 11)) / 100)
                if hp == ap: hp += 1
                g = {"id": gid, "date": day, "datetime": day + "T23:30:00Z", "season": season, "status": "Final",
                     "postseason": False, "home_team": {"id": h + 1, "abbreviation": ABBR[h]},
                     "visitor_team": {"id": a + 1, "abbreviation": ABBR[a]},
                     "home_team_score": hp, "visitor_team_score": ap}
                games.append(g)
                for team, pts in ((h, hp), (a, ap)):
                    fga = round(poss * 0.88); fta = round(poss * 0.25)
                    stats.append({"min": "240:00", "pts": pts, "fga": fga, "fgm": round(fga * 0.47), "fg3a": round(fga * 0.4),
                                  "fg3m": round(fga * 0.145), "fta": fta, "ftm": round(fta * 0.78), "oreb": 10, "dreb": 33,
                                  "turnover": round(poss - fga + 10 - 0.44 * fta), "team": {"id": team + 1}, "game": {"id": gid}})
                # A market that knows the true ratings (efficient line), standard -110 both sides
                true_margin = exp_poss * (2.5 + off[h] - dfn[a] - off[a] + dfn[h]) / 100
                true_total = exp_poss * (228 + 2.5 + off[h] - dfn[a] + off[a] - dfn[h]) / 100
                odds.append({"id": gid, "game_id": gid, "vendor": "draftkings", "opened_at": day + "T15:00:00Z",
                             "spread_home_value": str(-round(true_margin * 2) / 2), "spread_home_odds": -110,
                             "spread_away_odds": -110, "total_value": str(round(true_total * 2) / 2),
                             "total_over_odds": -110, "total_under_odds": -110})
    return games, stats, odds, (off, dfn, pace)


def test_possessions_formula():
    assert possessions({"fga": 88, "oreb": 10, "turnover": 13, "fta": 25}) == pytest.approx(88 - 10 + 13 + 11)


def test_ratings_recover_true_strength_and_use_only_past():
    games, stats, _, (off, dfn, _) = synth_league()
    recs = game_records(games, stats)
    model = RatingModel(half_life_days=400).fit(recs, "2026-12-31")
    est = [model.rating(str(i + 1))[0] + model.rating(str(i + 1))[1] for i in range(len(off))]
    true = [o + d for o, d in zip(off, dfn)]
    corr = sum((e - sum(est) / 8) * (t - sum(true) / 8) for e, t in zip(est, true))
    assert corr > 0  # net ratings line up with truth
    projs = pregame_projections(games[:200], stats)
    first_day = projs[0]["date"]
    assert all(r["date"] >= first_day for r in projs)
    # changing a FUTURE game's score must not change an earlier projection
    altered = [dict(g) for g in games[:200]]
    altered[-1]["home_team_score"] += 50
    again = {p["game_id"]: p["proj_margin"] for p in pregame_projections(altered, stats)}
    for p in projs:
        if p["date"] < altered[-1]["date"]:
            assert again[p["game_id"]] == pytest.approx(p["proj_margin"])


def test_schedule_travel():
    games = [{"id": 1, "date": "2026-01-01", "home_team": {"id": 1, "abbreviation": "BOS"}, "visitor_team": {"id": 2, "abbreviation": "DEN"}},
             {"id": 2, "date": "2026-01-02", "home_team": {"id": 2, "abbreviation": "DEN"}, "visitor_team": {"id": 1, "abbreviation": "BOS"}}]
    f = schedule_travel_features(games)
    assert f["2"]["away_b2b"] == 1 and f["2"]["home_b2b"] == 1
    assert 2700 < f["2"]["away_travel_km"] < 3100
    assert f["2"]["away_at_altitude"] == 1 and f["2"]["away_tz_shift"] == -2


def test_normal_sides_push():
    over, push = normal_sides(0.0, 0.0, 10.0, True)
    assert over == pytest.approx((1 - push) / 2) and 0.03 < push < 0.05


def test_efficient_market_gives_no_profitable_game_bets():
    games, stats, odds, _ = synth_league(1)
    rows, cov = build_rows(games, stats, odds)
    cands, report = run_games(rows)
    assert cands and report["markets"]["spread"]["out_of_sample_mse"]
    scored, _ = walk_forward_score([c for c in cands if c["model_version"] == "residual_v2"])
    chosen = select(scored, 0.0)
    if len(chosen) >= 50:
        res = simulate(chosen, "flat")
        assert res["roi_on_turnover_pct"] < 5  # market knew the truth; no durable edge


def _cand(i, p_model, win, price=-110, opp=-110, market="m", day=None, game=None, player=None):
    return {"candidate_id": str(i), "season": 2025, "date": day or f"2026-0{1 + i % 6}-{1 + i % 27:02d}",
            "game_id": game or str(i), "player_id": player, "market": market, "side": "over", "line": 1.5,
            "price": price, "opp_price": opp, "p_model": p_model, "outcome": "win" if win else "loss",
            "model_version": "v"}


def test_calibrator_learns_informative_vs_useless_model():
    rng = random.Random(3)
    useful, useless = [], []
    for i in range(4000):
        truth = rng.uniform(0.3, 0.7)
        win = rng.random() < truth
        useful.append(_cand(i, truth, win))
        useless.append(_cand(i, rng.uniform(0.3, 0.7), win))
    a = Calibrator().fit(prepare(useful)).describe()
    b = Calibrator().fit(prepare(useless)).describe()
    assert a["b_model"] > 0.6 and abs(b["b_model"]) < 0.25


def test_select_respects_limits_and_kelly_tiers():
    day = "2026-01-05"
    scored = []
    for i in range(10):
        c = _cand(i, 0.6, True, day=day, game="G1", player="P1" if i < 5 else None)
        c.update(p=0.62, p_lo=0.60, ev=0.1, ev_lo=0.08 - i * 0.001, p_market=0.5, p_push=0.0, market=f"m{i}")
        scored.append(c)
    chosen = select(scored, 0.0)
    assert len(chosen) <= 2  # two per game
    assert sum(c.get("player_id") == "P1" for c in chosen) <= 1
    stake, tier = kelly_tier({"price": -110, "p_lo": 0.59})
    assert tier == "A++" and stake == 0.03
    assert kelly_tier({"price": -110, "p_lo": 0.53})[1] == "pass"


def test_tau_targets_count():
    scored = []
    for i in range(300):
        c = _cand(i, 0.55, i % 2 == 0, day=f"2026-01-{1 + i % 28:02d}", game=str(i))
        c.update(p=0.56, p_lo=0.54, ev=0.05, ev_lo=0.001 * i, p_market=0.5, p_push=0.0)
        scored.append(c)
    tau = tau_for_target(scored, 100)
    assert len(select(scored, tau)) <= 100


def test_rank_and_stake_end_to_end_split_date():
    rng = random.Random(5)
    cands = []
    for i in range(6000):
        day = f"2026-{1 + (i // 1000):02d}-{1 + (i % 28):02d}"
        truth = rng.uniform(0.35, 0.65)
        win = rng.random() < truth
        for side, p, w in (("over", truth, win), ("under", 1 - truth, not win)):
            c = _cand(i, min(max(p + rng.gauss(0, 0.03), 0.01), 0.99), w, day=day, game=f"g{i}")
            c.update(side=side, candidate_id=f"{i}{side}")
            cands.append(c)
    report = rank_run(cands, split_date="2026-04-01", targets=(50, 100), mc_paths=200)
    assert set(report["sweep"]) == {"50", "100"}
    assert report["reliability_test"]["m"]["n"] > 0


def test_planted_back_to_back_inefficiency_is_found_out_of_sample():
    games, stats, odds, _ = synth_league(1, b2b_penalty=8.0)
    rows, _ = build_rows(games, stats, odds)
    cands, _ = run_games(rows)
    scored, _ = walk_forward_score([c for c in cands if c["model_version"] == "residual_v2"])
    res = simulate(select(scored, 0.0), "flat")
    assert res["bets"] >= 100 and res["roi_on_turnover_pct"] > 5


def test_prop_candidate_export_is_out_of_sample_and_two_sided():
    from tests.test_pricing_and_v2 import synthetic_season
    from scripts.export_prop_candidates import export
    feats, lines, games = synthetic_season(4, priced=True)
    cands, coverage = export([{"season": 2025, "features": feats, "lines": lines, "games": games}])
    assert cands and all(c["market"] == "prop_threes" for c in cands)
    first_month = min(c["date"][:7] for c in cands)
    assert first_month > min(f["game_date"][:7] for f in feats)  # burn-in month never scored in-sample
    by_bet = {}
    for c in cands:
        by_bet.setdefault(c["candidate_id"].rsplit(":", 1)[0], []).append(c)
    assert all(len(v) == 2 for v in by_bet.values())
    for v in by_bet.values():
        assert sum(x["outcome"] == "win" for x in v) <= 1
    scored, _ = walk_forward_score(cands)
    res = simulate(select(scored, 0.0), "flat")
    assert res["bets"] == 0 or res["roi_on_turnover_pct"] < 10  # fairly priced synthetic market


def test_pooled_grouping_and_staking_variants():
    from v13.ranking import stake_fraction, parametric_monte_carlo
    rng = random.Random(9)
    cands = []
    for i in range(3000):
        truth = rng.uniform(0.35, 0.65); win = rng.random() < truth
        for mkt in ("a", "b"):
            c = _cand(i, truth, win, day=f"2026-{1 + i // 600:02d}-{1 + i % 28:02d}", game=f"g{i}{mkt}", market=mkt)
            c["candidate_id"] = f"{i}{mkt}"
            cands.append(c)
    pooled, fits = walk_forward_score(cands, grouping="pooled")
    assert pooled and all("pooled|all" in f or not f for f in fits.values())
    b = {"tier_stake": 0.03}
    assert stake_fraction(b, "flat") == 0.01 and stake_fraction(b, "capped") == 0.02
    assert stake_fraction(b, "half") == pytest.approx(0.015)
    chosen = select(pooled, -1.0)[:200]
    for c in chosen:
        c["p_market"] = 0.5238 if c.get("p_market") is None else c["p_market"]
    zero = parametric_monte_carlo(chosen, 0.0, "flat", paths=300)
    assert zero["prob_below_start"] > 0.5  # with no real edge, the vig wins


def test_feasibility_zero_edge_loses_and_big_edge_wins():
    from scripts.bankroll_feasibility import simulate as fsim
    none = fsim(200, [(-110, -110)], [(0.01, 1.0)], 0.0, paths=2000)
    big = fsim(200, [(-110, -110)], [(0.02, 1.0)], 0.10, paths=2000)
    assert none["median_end"] < 10000 and none["mean_roi_pct"] < 0
    assert big["prob_ge_15000"] > 0.5
