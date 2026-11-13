"""Round 9 end-to-end acceptance run (offline, synthetic fixtures; no API key needed).

Demonstrates, and fails the run if any of these does not hold:
  1. invalid historical prices are quarantined with explicit reasons and never silently dropped;
  2. a validated pipeline refuses unstamped or contaminated odds (fails closed);
  3. valid prospective logger observations are preserved with full provenance and stable ids;
  4. the prospective ledger rejects retroactive, PASS-tier and backtest entries, and refuses to
     report a file that mixes shadow and actual records.
python -m scripts.acceptance_round9 --out research_results/acceptance/round9.json
"""
import argparse, json, tempfile
from datetime import datetime, timezone
from pathlib import Path

from v13 import ledger
from v13.provenance import UnverifiedOddsError, quarantine, require_eligible

GAMES = [{"id": 1, "season": 2024, "datetime": "2025-01-10T00:30:00Z"},
         {"id": 2, "season": 2025, "datetime": "2026-01-10T00:30:00Z"},
         {"id": 3, "season": 2025, "datetime": "2026-01-11T00:30:00Z"},
         {"id": 4, "season": 2025}]
ROWS = [{"game_id": 1, "opened_at": "2025-01-09T15:00:00Z", "tag": "outside_coverage"},
        {"game_id": 2, "opened_at": "2026-01-09T15:00:00Z", "tag": "valid"},
        {"game_id": 3, "opened_at": "2026-01-11T00:30:00Z", "tag": "at_tip"},
        {"game_id": 3, "tag": "no_timestamp"},
        {"game_id": 4, "opened_at": "2026-01-09T15:00:00Z", "tag": "no_tip"},
        {"game_id": 99, "opened_at": "2026-01-09T15:00:00Z", "tag": "unknown_game"}]


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code, self._p, self.headers = status, payload, {}
        self.text = json.dumps(payload)
    def json(self): return json.loads(self.text)
    def raise_for_status(self): pass


class _Session:
    def get(self, url, params=None, headers=None, timeout=None):
        if url.endswith("/v1/games"):
            return _Resp({"data": [{"id": 7, "datetime": "2026-10-28T23:30:00Z"}], "meta": {}})
        if url.endswith("/nba/v2/odds"):
            return _Resp({"data": [{"game_id": 7, "vendor": "draftkings", "updated_at": "2026-10-28T17:55:00Z",
                                    "spread_home_value": "-4.5", "spread_home_odds": -110, "spread_away_value": "4.5",
                                    "spread_away_odds": -110, "total_value": "224.5", "total_over_odds": -108,
                                    "total_under_odds": -112}], "meta": {}})
        return _Resp({"data": [{"game_id": 7, "player_id": 11, "vendor": "fanduel", "prop_type": "rebounds",
                                "line_value": "8.5", "updated_at": "2026-10-28T17:50:00Z",
                                "market": {"type": "over_under", "over_odds": -120, "under_odds": 100}}]})


def run():
    checks = {}
    elig, excl, summ = quarantine(ROWS, GAMES, "bdl_opening_player_props")
    checks["only_documented_pretip_row_eligible"] = [r["tag"] for r in elig] == ["valid"]
    checks["every_excluded_row_kept_with_reason"] = len(excl) == len(ROWS) - 1 and all(r["_prov"]["reason"] != "ok" for r in excl)
    checks["reasons"] = summ["reasons"]
    for name, rows in (("raw_unstamped_rows_rejected", [{k: v for k, v in r.items()} for r in ROWS]),
                       ("contaminated_stamped_rows_rejected", excl)):
        try:
            require_eligible(rows); checks[name] = False
        except UnverifiedOddsError:
            checks[name] = True
    checks["eligible_rows_accepted"] = bool(require_eligible(elig))
    _, _, unknown = quarantine(ROWS[:1], GAMES, "some_unregistered_feed")
    checks["unknown_source_fails_closed"] = unknown["eligible"] == 0 and unknown["reasons"] == {"unknown_source": 1}

    from scripts.log_live_odds import snapshot
    now = datetime(2026, 10, 28, 18, 0, tzinfo=timezone.utc)
    obs, raw, man = snapshot("k", now=now, session=_Session(), commit="test")
    again, _, _ = snapshot("k", now=now, session=_Session(), commit="test")
    needed = {"observation_id", "quote_id", "captured_at_utc", "vendor_updated_at", "game_id", "tip_utc",
              "sportsbook", "market", "side", "line", "american_odds", "source_endpoint", "response_sha256",
              "logger_version", "git_commit"}
    checks["logger_observations_have_provenance"] = len(obs) == 6 and all(needed <= set(o) for o in obs)
    checks["logger_ids_deterministic"] = [o["observation_id"] for o in obs] == [o["observation_id"] for o in again]
    checks["logger_raw_responses_hashed"] = all(len(r["sha256"]) == 64 for r in raw) and man["requests_used"] == 3

    tmp = Path(tempfile.mkdtemp())
    pred = {"prediction_id": "p1", "predicted_at_utc": "2026-10-28T18:00:00Z", "tip_utc": "2026-10-28T23:30:00Z",
            "game_id": 7, "market": "prop_rebounds", "side": "over", "line": 8.5, "price": -120, "opp_price": 100,
            "sportsbook": "fanduel", "p_model": 0.58, "p_market": 0.53, "ev": 0.04, "tier": "A",
            "stake_pct_of_bankroll": 0.01, "model_version": "v", "git_commit": "test", "source": "live_logger"}
    ledger.append_prediction(tmp / "s.jsonl", pred, "shadow", now=now)
    def rejected(**kw):
        try:
            ledger.append_prediction(tmp / "s.jsonl", {**pred, **kw}, "shadow", now=now)
            return False
        except ValueError:
            return True
    checks["ledger_rejects_pass_tier"] = rejected(prediction_id="p2", tier="pass")
    checks["ledger_rejects_backtest_source"] = rejected(prediction_id="p3", source="backtest_2025")
    try:
        ledger.append_prediction(tmp / "s.jsonl", {**pred, "prediction_id": "p4"}, "shadow",
                                 now=datetime(2026, 10, 29, 1, tzinfo=timezone.utc))
        checks["ledger_rejects_retroactive_entry"] = False
    except ValueError:
        checks["ledger_rejects_retroactive_entry"] = True
    mixed = tmp / "m.jsonl"
    ledger.append_prediction(mixed, pred, "shadow", now=now)
    ledger.append_prediction(mixed, {**pred, "prediction_id": "p9", "confirmed_price": -120, "stake_amount": 100.0,
                                     "bet_reference": "t"}, "actual", now=now)
    try:
        ledger.report(mixed); checks["mixed_ledger_report_refused"] = False
    except ValueError:
        checks["mixed_ledger_report_refused"] = True
    checks["shadow_report_labelled"] = ledger.report(tmp / "s.jsonl")["ledger"].startswith("SHADOW")
    passed = all(v for k, v in checks.items() if isinstance(v, bool))
    return {"passed": passed, "checks": checks}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); a = p.parse_args()
    res = run()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    if not res["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
