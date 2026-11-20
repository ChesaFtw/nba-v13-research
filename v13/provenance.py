"""Historical odds eligibility gate (quarantine) with explicit provenance. Fails closed.

A price may enter a VALIDATED betting backtest or report only if:
  1. its source is registered below (unknown source -> excluded),
  2. the game's season is inside that source's documented coverage,
  3. the source's quote timestamp field is present and parseable,
  4. the game's scheduled tip time is known, and
  5. the quote timestamp is strictly before tip (at/after tip -> excluded).
A field named "opening" is never evidence of pre-tip availability by itself.

Excluded rows are preserved (with an exclusion reason) for forensics; they are never deleted
and never silently dropped into the eligible set. Eligibility depends on the source, so an
independently verified source can be registered with its own documented coverage.
"""
from collections import Counter
from datetime import datetime, timezone

from v13.pricing import parse_utc

GATE_VERSION = "odds-gate-1"

# Documented coverage, per source. Seasons are BALLDONTLIE season labels (2025 = 2025-26).
SOURCES = {
    # OpenAPI: opening endpoints cover "the most recently completed season and ongoing seasons
    # where available" -> as of 2026-10 that is 2025-26 and 2026-27. Forensics (run 38652187615)
    # found 2023-24 and 2024-25 opening timestamps set exactly at tip.
    "bdl_opening_game_odds": {"timestamp_field": "opened_at", "documented_seasons": {2025, 2026},
                              "evidence": "BALLDONTLIE OpenAPI /nba/v2/odds/opening; forensics run 38652187615"},
    "bdl_opening_player_props": {"timestamp_field": "opened_at", "documented_seasons": {2025, 2026},
                                 "evidence": "BALLDONTLIE OpenAPI /nba/v2/odds/player_props/opening"},
    # Our own logger: the capture time is ours, so any captured season counts.
    "v13_live_logger": {"timestamp_field": "captured_at_utc", "documented_seasons": None,
                        "evidence": "scripts/log_live_odds.py append-only snapshots"},
}

REASONS = ("ok", "unknown_source", "unknown_season", "season_outside_documented_coverage",
           "missing_quote_timestamp", "missing_tip_time", "quote_at_or_after_tip")


def assess(row, source, tip_utc, season):
    """Return (eligible, reason) for one odds row."""
    spec = SOURCES.get(source)
    if spec is None:
        return False, "unknown_source"
    if season is None:
        return False, "unknown_season"
    if spec["documented_seasons"] is not None and int(season) not in spec["documented_seasons"]:
        return False, "season_outside_documented_coverage"
    ts = parse_utc(row.get(spec["timestamp_field"]))
    if ts is None:
        return False, "missing_quote_timestamp"
    tip = tip_utc if isinstance(tip_utc, datetime) else parse_utc(tip_utc)
    if tip is None:
        return False, "missing_tip_time"
    if ts >= tip:
        return False, "quote_at_or_after_tip"
    return True, "ok"


def game_index(games):
    """game_id -> (tip datetime or None, season or None) from a BALLDONTLIE game index."""
    return {str(g["id"]): (parse_utc(g.get("datetime")), g.get("season")) for g in games if g.get("id") is not None}


def quarantine(rows, games, source, now=None):
    """Split rows into (eligible, excluded, summary). Every row gets a `_prov` stamp."""
    idx = game_index(games)
    checked = (now or datetime.now(timezone.utc)).isoformat()
    eligible, excluded, reasons = [], [], Counter()
    for r in rows:
        tip, season = idx.get(str(r.get("game_id")), (None, None))
        ok, reason = assess(r, source, tip, season)
        stamped = {**r, "_prov": {"source": source, "season": season, "tip_utc": tip.isoformat() if tip else None,
                                  "timestamp_field": SOURCES.get(source, {}).get("timestamp_field"),
                                  "eligible": ok, "reason": reason, "gate_version": GATE_VERSION,
                                  "checked_at_utc": checked}}
        (eligible if ok else excluded).append(stamped)
        reasons[reason] += 1
    summary = {"source": source, "gate_version": GATE_VERSION, "rows": len(rows), "eligible": len(eligible),
               "excluded": len(excluded), "reasons": dict(reasons),
               "documented_seasons": sorted(SOURCES[source]["documented_seasons"])
               if source in SOURCES and SOURCES[source]["documented_seasons"] else "all captured"}
    return eligible, excluded, summary


class UnverifiedOddsError(ValueError):
    pass


def require_eligible(rows, context="validated report"):
    """Fail closed unless EVERY row carries a passing gate stamp."""
    bad = [r for r in rows if not ((r.get("_prov") or {}).get("eligible") is True
                                   and (r.get("_prov") or {}).get("gate_version") == GATE_VERSION)]
    if bad:
        reasons = Counter((r.get("_prov") or {}).get("reason", "no_provenance_stamp") for r in bad)
        raise UnverifiedOddsError(f"{len(bad)} odds rows lack a passing {GATE_VERSION} stamp for {context}: {dict(reasons)}")
    return rows


def evidence_class(validated):
    return "validated" if validated else "exploratory_unverified_prices"
