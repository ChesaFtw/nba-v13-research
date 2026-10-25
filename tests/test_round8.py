import json
import random

import pytest

from scripts.odds_forensics import classify, rows_for_game, sample_games, summarize
from v13 import ledger
from v13.ranking import select, tau_for_target, walk_forward_score


def test_forensic_classification():
    assert classify(-5.0) == "pregame"
    assert classify(1.5) == "in_game_window"
    assert classify(5.0) == "exact_hour_shift"      # whole-hour offset after tip: possible timezone bug
    assert classify(5.4) == "same_or_next_day"
    assert classify(400.0) == "days_or_more_after"  # ingestion/backfill-like
    assert classify(None) == "missing_or_unparseable"


def test_forensic_rows_and_eligibility():
    g = {"id": 9, "date": "2024-01-02", "datetime": "2024-01-03T00:30:00Z"}
    fetched = {"opening": ([{"vendor": "dk", "opened_at": "2024-01-02T15:00:00Z", "spread_home_value": "-3"},
                            {"vendor": "fd", "opened_at": "2025-06-01T00:00:00Z"}], "ok"),
               "current": ([], "ok")}
    rows = rows_for_game(g, 2024, fetched, documented={2025})
    opening = [r for r in rows if r["endpoint"] == "opening"]
    assert [r["classification"] for r in opening] == ["pregame", "days_or_more_after"]
    assert not any(r["pregame_eligible"] for r in rows)  # 2024 outside documented coverage
    rows25 = rows_for_game(g, 2025, fetched, documented={2025})
    assert sum(r["pregame_eligible"] for r in rows25) == 1
    assert summarize(rows25)["2025|opening"]["pregame_eligible_rows"] == 1


def test_sample_games_is_deterministic_and_spread_over_months():
    games = [{"id": i, "date": f"2025-{1 + i % 4:02d}-{1 + i % 27:02d}", "postseason": False, "status": "Final"} for i in range(200)]
    a, b = sample_games(games, 8), sample_games(games, 8)
    assert [g["id"] for g in a] == [g["id"] for g in b]
    assert len({g["date"][:7] for g in a}) == 4


def _scored(i, ev_lo, p_lo, day="2026-01-05"):
    return {"candidate_id": str(i), "date": day, "game_id": f"g{i}", "player_id": None, "market": "m",
            "price": -110, "p": p_lo + 0.01, "p_lo": p_lo, "ev": ev_lo + 0.01, "ev_lo": ev_lo, "p_market": 0.5,
            "outcome": "win", "season": 2025}


def test_pass_tier_and_nonpositive_ev_are_never_selected():
    rows = [_scored(1, 0.05, 0.56), _scored(2, 0.004, 0.528), _scored(3, -0.01, 0.52), _scored(4, 0.0, 0.5238)]
    chosen = select(rows, -1.0)
    assert [c["candidate_id"] for c in chosen] == ["1"]
    assert all(c["tier"] in ("A", "A+", "A++") for c in chosen)
    assert tau_for_target(rows, 10) >= 0.0


def test_walk_forward_fit_never_sees_scored_period():
    rng = random.Random(2)
    cands = []
    for i in range(3000):
        truth = rng.uniform(0.4, 0.6); w = rng.random() < truth
        cands.append({"candidate_id": str(i), "season": 2025, "date": f"2026-{1 + i // 600:02d}-{1 + i % 28:02d}",
                      "game_id": str(i), "player_id": None, "market": "m", "side": "over", "line": 1.5, "price": -110,
                      "opp_price": -110, "p_model": truth, "outcome": "win" if w else "loss", "model_version": "v"})
    _, fits = walk_forward_score(cands)
    for per, f in fits.items():
        for v in f.values():
            if v.get("train_max_date"):
                assert v["train_max_date"] < v["scored_period_start"]


def test_rank_and_stake_rejects_overlapping_windows():
    from scripts.rank_and_stake import run
    rng = random.Random(4)
    cands = []
    for i in range(2000):
        truth = rng.uniform(0.4, 0.6); w = rng.random() < truth
        cands.append({"candidate_id": str(i), "season": 2025 if i % 2 else 2024, "date": f"2026-{1 + i // 400:02d}-{1 + i % 28:02d}",
                      "game_id": str(i), "player_id": None, "market": "m", "side": "over", "line": 1.5, "price": -110,
                      "opp_price": -110, "p_model": truth, "outcome": "win" if w else "loss", "model_version": "v"})
    with pytest.raises(AssertionError):
        run(cands, validation_season=2024, test_season=2025, targets=(10,), mc_paths=50)


PRED = {"prediction_id": "p1", "predicted_at_utc": "2026-11-01T18:00:00Z", "tip_utc": "2026-11-02T00:00:00Z",
        "game_id": 1, "market": "prop_rebounds", "side": "over", "line": 8.5, "price": -110, "opp_price": -110,
        "sportsbook": "draftkings", "p_model": 0.56, "p_market": 0.5, "ev": 0.07, "tier": "A",
        "stake_pct_of_bankroll": 0.01, "model_version": "v", "git_commit": "abc", "source": "live_logger"}


def test_ledger_chain_rules_and_tamper_detection(tmp_path):
    path = tmp_path / "shadow.jsonl"
    from datetime import datetime, timezone
    now = datetime(2026, 11, 1, 19, tzinfo=timezone.utc)
    ledger.append_prediction(path, PRED, "shadow", now=now)
    with pytest.raises(ValueError):
        ledger.append_prediction(path, {**PRED, "prediction_id": "p2", "tier": "pass"}, "shadow", now=now)
    with pytest.raises(ValueError):  # after tip
        ledger.append_prediction(path, {**PRED, "prediction_id": "p3", "predicted_at_utc": "2026-11-02T01:00:00Z"}, "shadow",
                                 now=datetime(2026, 11, 2, 2, tzinfo=timezone.utc))
    with pytest.raises(ValueError):  # shadow cannot claim a placed bet
        ledger.append_prediction(path, {**PRED, "prediction_id": "p4", "confirmed_price": -110, "stake_amount": 100,
                                        "bet_reference": "x"}, "shadow", now=now)
    ledger.append_close(path, "p1", 8.5, -125, 105, "2026-11-01T23:55:00Z", "live_logger")
    ledger.append_settlement(path, "p1", "win", "2026-11-02T03:00:00Z", "box_score")
    rep = ledger.report(path)
    assert rep["ledger"].startswith("SHADOW") and rep["wins"] == 1 and rep["clv_same_line"]["mean_prob_pts"] > 0
    assert ledger.verify(path) == 3
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0]); rec["price"] = 150; lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        ledger.verify(path)


def test_actual_ledger_requires_confirmation(tmp_path):
    from datetime import datetime, timezone
    now = datetime(2026, 11, 1, 19, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        ledger.append_prediction(tmp_path / "actual.jsonl", PRED, "actual", now=now)
    ok = ledger.append_prediction(tmp_path / "actual.jsonl", {**PRED, "confirmed_price": -115, "stake_amount": 100.0,
                                                               "bet_reference": "ticket-1"}, "actual", now=now)
    assert ok["ledger_kind"] == "actual"
