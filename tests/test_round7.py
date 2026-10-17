from scripts.odds_coverage_audit import audit_season
from scripts.rank_and_stake import run as rank_run


def test_coverage_audit_separates_causes():
    games = [{"id": 1, "datetime": "2025-01-02T00:00:00Z", "postseason": False},
             {"id": 2, "datetime": "2025-01-03T00:00:00Z", "postseason": False},
             {"id": 3, "postseason": False},  # no tip time
             {"id": 4, "datetime": "2025-01-05T00:00:00Z", "postseason": False}]
    rows = [{"game_id": 1, "opened_at": "2025-01-01T12:00:00Z", "spread_home_odds": -110, "spread_away_odds": -110,
             "total_over_odds": -110, "total_under_odds": -110, "vendor": "dk"},
            {"game_id": 2, "opened_at": "2025-01-03T01:00:00Z", "vendor": "dk"},
            {"game_id": 3, "opened_at": "2025-01-03T12:00:00Z", "vendor": "dk"}]
    r = audit_season(games, rows)
    assert r["games_with_pregame_row"] == 1 and r["games_with_only_post_tip"] == 1
    assert r["games_without_tip_time"] == 1 and r["games_with_only_unverifiable"] == 1
    assert r["games_without_any_row"] == 1 and r["rows_with_spread_price"] == 1


def test_ranking_reports_untestable_instead_of_crashing_or_relabelling():
    cands = [{"candidate_id": str(i), "season": 2025, "date": f"2026-0{1 + i % 5}-{1 + i % 27:02d}", "game_id": str(i),
              "player_id": None, "market": "spread", "side": "home", "line": -3.5, "price": -110, "opp_price": -110,
              "p_model": 0.5, "outcome": "win" if i % 2 else "loss", "model_version": "residual_v2"} for i in range(600)]
    rep = rank_run(cands, validation_season=2024, test_season=2025)
    assert rep["status"] == "UNTESTABLE" and rep["scored_validation"] == 0
    assert rep["raw_candidates_by_season"] == {"2025": 600}


def test_prereg_v2_market_verdict_on_fair_market():
    from tests.test_pricing_and_v2 import synthetic_season
    from scripts.prereg_v2_market import run
    feats, lines, games = synthetic_season(8, priced=True)
    out = run(feats, lines, games, "threes", "edge")
    assert out["verdict"] in ("FAIL", "UNTESTABLE")
    if out["verdict"] == "FAIL":
        assert "power" in out["primary"] and out["priced_share"] >= 0.9
