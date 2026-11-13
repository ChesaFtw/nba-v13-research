import math
import random

import pytest

from v13.pricing import (implied_probability, no_vig_pair, pregame_status, profit_per_unit,
                         prop_prices, tip_times, wilson, game_price)
from scripts.backtest_prop_holdout_v2 import evaluate, p_over, simulate, normalize_lines
from scripts.audit_prop_ledger import audit
from scripts.holdout_spreads_totals import make, sim
from collections import Counter


def test_price_math():
    assert profit_per_unit(-110) == pytest.approx(100 / 110)
    assert profit_per_unit(150) == pytest.approx(1.5)
    assert implied_probability(-150) == pytest.approx(0.6)
    a, b = no_vig_pair(-150, 120)
    assert a + b == pytest.approx(1.0) and a > b
    with pytest.raises(ValueError):
        profit_per_unit(-50)


def test_prop_prices_and_missing():
    assert prop_prices({"market": {"type": "over_under", "over_odds": -140, "under_odds": 110}}) == (-140, 110)
    assert prop_prices({"market": {"type": "over_under"}}) == (None, None)
    assert prop_prices({"market": {"type": "milestone", "odds": 200}}) == (None, None)
    assert game_price({"total_over_odds": "-115"}, "total", "over") == -115


def test_pregame_status():
    tips = tip_times([{"id": 7, "datetime": "2026-01-05T00:30:00Z"}])
    assert pregame_status("2026-01-04T18:00:00Z", tips["7"]) == "pregame"
    assert pregame_status("2026-01-05T00:30:00Z", tips["7"]) == "post_tip"  # at tip is not pregame
    assert pregame_status("2026-01-05T01:00:00Z", tips["7"]) == "post_tip"
    assert pregame_status(None, tips["7"]) == "unverifiable"
    assert pregame_status("2026-01-04T18:00:00Z", None) == "unverifiable"


def test_wilson_bounds():
    lo, hi = wilson(599, 986)
    assert 0.57 < lo < 0.58 and 0.63 < hi < 0.64


def test_poisson_p_over():
    # P(X > 1.5) for Poisson(2) = 1 - e^-2 (1 + 2)
    assert p_over(2.0, 1.5, "threes", 1) == pytest.approx(1 - math.exp(-2) * 3)


def line_row(gid, pid, line, opened, over=-110, under=-110, vendor="caesars", prop="threes", rid=1):
    return {"id": rid, "game_id": gid, "player_id": pid, "vendor": vendor, "prop_type": prop,
            "line_value": str(line), "opened_at": opened,
            "market": {"type": "over_under", "over_odds": over, "under_odds": under}}


def test_normalize_drops_post_tip_and_flags_conflicts():
    tips = tip_times([{"id": 1, "datetime": "2026-01-05T00:00:00Z"}])
    cov = Counter()
    rows = [line_row(1, 9, 2.5, "2026-01-05T01:00:00Z", rid=1),  # after tip -> dropped
            line_row(1, 9, 1.5, "2026-01-04T15:00:00Z", rid=2),
            line_row(1, 9, 2.5, "2026-01-04T16:00:00Z", rid=3)]  # same vendor, different value
    chosen = normalize_lines(rows, tips, cov)
    assert chosen[("1", "9", "threes")][1] == 1.5  # earliest pregame row
    assert cov["line_post_tip"] == 1 and cov["conflicting_line_values"] == 1


def fair_american(p, vig=0.045):
    """American price for true probability p with proportional vig."""
    q = min(p * (1 + vig / 2), 0.97)
    return int(round(-100 * q / (1 - q))) if q >= 0.5 else int(round(100 * (1 - q) / q))


def synthetic_season(seed, n_days=120, players=40, juice_under=-110, priced=False):
    """Players whose threes ~ Poisson(true rate); line = floor(rate)+0.5.

    priced=False: both sides -110 (what a -110 backtest implicitly assumes).
    priced=True: each side priced from the TRUE probability plus vig, like a sharp book.
    """
    rng = random.Random(seed)
    rates = [rng.uniform(0.8, 3.5) for _ in range(players)]
    games, features, lines = [], [], []
    hist = {p: [] for p in range(players)}
    gid = 0
    for d in range(n_days):
        day = f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}"
        gid += 1
        games.append({"id": gid, "datetime": day + "T23:00:00Z"})
        for p in range(players):
            # Knuth Poisson draw
            L, k, prod = math.exp(-rates[p]), 0, rng.random()
            while prod > L:
                k += 1; prod *= rng.random()
            prior = hist[p]
            if len(prior) >= 20:
                f = {"player_id": p, "game_id": gid, "game_date": day, "history_games": len(prior)}
                for m in ("points", "rebounds", "assists", "threes", "pra"):
                    for w in (5, 10, 20):
                        f[f"{m}_avg_{w}"] = sum(prior[-w:]) / w
                    f[f"{m}_actual"] = k
                features.append(f)
                line = math.floor(rates[p]) + 0.5
                over, under = -110, juice_under
                if priced:
                    po = p_over(rates[p], line, "threes", 1)
                    over, under = fair_american(po), fair_american(1 - po)
                lines.append(line_row(gid, p, line, day + "T15:00:00Z", over=over, under=under, rid=len(lines)))
            hist[p].append(k)
    return features, lines, games


def test_symmetric_110_grading_manufactures_a_60pct_record():
    """Mechanism check: half-point count lines graded at -110 on both sides look like 60%."""
    features, lines, games = synthetic_season(1)
    held = evaluate(features, lines, games, "actual")["markets"]["threes"]["edge"]["holdout"]
    assert held["hit_rate"] > 0.57


def test_v2_finds_no_profit_when_prices_reflect_true_probability():
    features, lines, games = synthetic_season(1, priced=True)
    threes = evaluate(features, lines, games, "actual")["markets"]["threes"]
    assert threes["status"] == "evaluated"
    for sel in ("edge", "ev"):
        held = threes[sel].get("holdout")
        if held and held["bets"] >= 100:
            assert held["roi_on_turnover_pct"] < 8
            assert held["vs_devigged_market"]["z"] < 2.5


def test_heavy_under_juice_is_charged_not_ignored():
    features, lines, games = synthetic_season(2, juice_under=-170)
    actual = evaluate(features, lines, games, "actual")["markets"]["threes"]["edge"]
    assumed = evaluate(features, lines, games, "assumed110")["markets"]["threes"]["edge"]
    if actual.get("holdout") and assumed.get("holdout"):
        a, b = actual["holdout"], assumed["holdout"]
        if a["by_side"]["under"] and a["bets"] == b["bets"]:
            assert a["roi_on_turnover_pct"] < b["roi_on_turnover_pct"]


def test_day_start_staking_is_independent_of_same_day_order():
    rows = [{"date": "2026-01-01", "game_id": str(i), "player_id": "1", "market": "threes", "line": 1.5,
             "actual": a, "x": [1, 1, 1], "over_odds": -110, "under_odds": -110, "vendor": "caesars",
             "opened_at": "x", "projection": 3.0} for i, a in enumerate([3, 0, 3])]
    led = simulate(rows, "edge", 0.5, "actual", 1.0)["ledger"]
    assert {x["stake"] for x in led} == {100.0}


def test_ledger_audit_detects_post_tip_and_reprices():
    games = [{"id": 1, "datetime": "2026-01-05T00:00:00Z"}, {"id": 2, "datetime": "2026-01-06T00:00:00Z"}]
    lines = [line_row(1, 9, 1.5, "2026-01-04T12:00:00Z", over=120, under=-160),
             line_row(2, 9, 1.5, "2026-01-06T02:00:00Z", over=120, under=-160, rid=2)]
    ledger = [{"date": "2026-01-04", "game_id": 1, "player_id": 9, "market": "threes", "line": 1.5,
               "projection": 0.8, "side": "under", "outcome": "win"},
              {"date": "2026-01-05", "game_id": 2, "player_id": 9, "market": "threes", "line": 1.5,
               "projection": 0.8, "side": "under", "outcome": "win"}]
    report = {"markets": {"threes": {"ridge_holdout_ledger": ledger}}}
    out = audit(report, lines, games, "threes")
    assert out["by_timing"]["post_tip"]["bets"] == 1
    assert out["pregame_verified_only"]["bets"] == 1
    o = out["overall"]
    assert o["roi_flat_at_actual_price_pct"] < o["roi_flat_at_assumed_110_pct"]


def test_spread_total_actual_prices_and_pregame_filter():
    preds = [{"game_id": 1, "date": "2026-01-04", "predicted_home_margin": 10, "actual_home_margin": 8,
              "predicted_total": 230, "actual_total": 210}]
    odds = [{"game_id": 1, "vendor": "draftkings", "opened_at": "2026-01-04T10:00:00Z",
             "spread_home_value": "-3.5", "spread_home_odds": -125, "spread_away_odds": 105,
             "total_value": "220.5", "total_over_odds": -105, "total_under_odds": -115}]
    tips = tip_times([{"id": 1, "datetime": "2026-01-05T00:00:00Z"}])
    rows = make(preds, odds, tips, "actual")
    spread = [r for r in rows if r["market"] == "spread"][0]
    total = [r for r in rows if r["market"] == "total"][0]
    assert spread["price"] == -125 and total["price"] == -105
    late = [dict(odds[0], opened_at="2026-01-05T01:00:00Z")]
    assert make(preds, late, tips, "actual") == []
    legacy = make(preds, odds)
    assert all(r["price"] == -110 for r in legacy)
    res = sim([dict(spread)], 1, True)
    assert res["wins"] == 1 and res["ending_bankroll"] == pytest.approx(10080.0)
