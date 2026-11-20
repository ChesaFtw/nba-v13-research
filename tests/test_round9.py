import pytest
from datetime import datetime, timezone

from v13.provenance import GATE_VERSION, UnverifiedOddsError, assess, quarantine, require_eligible
from v13.tier_validation import validate_tiers


def test_gate_reasons_and_fail_closed():
    tip = datetime(2026, 1, 10, 0, 30, tzinfo=timezone.utc)
    ok = {"opened_at": "2026-01-09T12:00:00Z"}
    assert assess(ok, "bdl_opening_player_props", tip, 2025) == (True, "ok")
    assert assess(ok, "bdl_opening_player_props", tip, 2024)[1] == "season_outside_documented_coverage"
    assert assess(ok, "bdl_opening_player_props", tip, None)[1] == "unknown_season"
    assert assess(ok, "mystery_feed", tip, 2025)[1] == "unknown_source"
    assert assess({"opened_at": "2026-01-10T00:30:00Z"}, "bdl_opening_game_odds", tip, 2025)[1] == "quote_at_or_after_tip"
    assert assess({}, "bdl_opening_game_odds", tip, 2025)[1] == "missing_quote_timestamp"
    assert assess(ok, "bdl_opening_game_odds", None, 2025)[1] == "missing_tip_time"
    # our own logger: capture time is authoritative for any season
    assert assess({"captured_at_utc": "2026-01-09T12:00:00Z"}, "v13_live_logger", tip, 2031) == (True, "ok")


def test_quarantine_preserves_everything_and_stamps():
    games = [{"id": 1, "season": 2025, "datetime": "2026-01-10T00:30:00Z"}]
    rows = [{"game_id": 1, "opened_at": "2026-01-09T12:00:00Z"}, {"game_id": 1, "opened_at": "2026-01-10T03:00:00Z"}]
    e, x, s = quarantine(rows, games, "bdl_opening_game_odds")
    assert len(e) == 1 and len(x) == 1 and s["rows"] == 2
    assert e[0]["_prov"]["gate_version"] == GATE_VERSION and x[0]["_prov"]["reason"] == "quote_at_or_after_tip"
    with pytest.raises(UnverifiedOddsError):
        require_eligible(e + x)
    with pytest.raises(UnverifiedOddsError):  # forged stamp from another gate version is refused
        require_eligible([{**e[0], "_prov": {**e[0]["_prov"], "gate_version": "old"}}])


def test_validated_backtest_refuses_unstamped_lines_and_labels_evidence():
    from tests.test_pricing_and_v2 import synthetic_season
    from scripts.backtest_prop_holdout_v2 import evaluate
    feats, lines, games = synthetic_season(3, priced=True)
    with pytest.raises(UnverifiedOddsError):
        evaluate(feats, lines, games, "actual", validated=True)
    for g in games:
        g["season"] = 2025
    elig, excl, _ = quarantine(lines, games, "bdl_opening_player_props")
    rep = evaluate(feats, elig, games, "actual", validated=True)
    assert rep["evidence_class"] == "validated" and not excl
    assert evaluate(feats, lines, games, "actual")["evidence_class"] == "exploratory_unverified_prices"


def test_rank_report_never_calls_mixed_evidence_validated():
    from scripts.rank_and_stake import run
    import random
    rng = random.Random(1); cands = []
    for i in range(3000):
        t = rng.uniform(0.4, 0.6); w = rng.random() < t
        cands.append({"candidate_id": str(i), "season": 2025, "date": f"2026-{1 + i // 600:02d}-{1 + i % 28:02d}",
                      "game_id": str(i), "player_id": None, "market": "m", "side": "over", "line": 1.5, "price": -110,
                      "opp_price": -110, "p_model": t, "outcome": "win" if w else "loss", "model_version": "v",
                      "evidence_class": "validated" if i % 2 else "exploratory_unverified_prices"})
    rep = run(cands, test_season=2025, split_date="2026-03-01", targets=(20,), mc_paths=50)
    assert rep["evidence_class"] == "exploratory_unverified_prices"


def _bet(tier, win, i, price=-110, pm=0.5):
    return {"tier": tier, "outcome": "win" if win else "loss", "price": price, "p_market": pm,
            "date": f"2026-{11 + i % 2:02d}-01", "game_id": str(i // 2)}


def test_tier_validation_orders_and_flags():
    bets = []
    for i in range(300):
        bets.append(_bet("A", i % 100 < 52, i))
        bets.append(_bet("A+", i % 100 < 56, 1000 + i))
        bets.append(_bet("A++", i % 100 < 60, 2000 + i))
    rep = validate_tiers(bets)
    assert rep["ordering"] == {"excess": "increasing", "roi_pct": "increasing"} and rep["verdict"] == "tiers_supported"
    small = validate_tiers(bets[:30])
    assert small["ordering"]["excess"] == "insufficient_sample" and small["verdict"] == "use_flat_staking"
    assert "multi_bet_games" in rep["correlation"] and rep["qualified_per_month"]


def test_acceptance_script_passes():
    from scripts.acceptance_round9 import run
    assert run()["passed"]
