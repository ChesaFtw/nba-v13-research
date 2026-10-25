"""Fail-closed research checks for historical point-in-time player availability.

This is a DATA GATE, not a fitted model. Box-score appearances cannot be
used as the universe of players predicted to play.
"""
from collections import Counter
from datetime import datetime, timezone

def parse_utc(value):
    if not isinstance(value, str):
        raise ValueError("timestamp required")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("timezone-aware timestamp required")
    return stamp.astimezone(timezone.utc)

def validate_appearance_universe(rows):
    """Require one pre-tip roster candidate per player/game, including DNPs.

    Each row: game_id, player_id, tipoff_at_utc, roster_observed_at_utc,
    appeared (bool), source. All players in the input are retained.
    Returns a coverage summary, never a betting-grade model score.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("no candidate rows")
    seen = set()
    games = set()
    outcomes = Counter()
    for row in rows:
        key = (row.get("game_id"), row.get("player_id"))
        if not all(key) or key in seen:
            raise ValueError("missing or duplicate game/player key")
        seen.add(key)
        if row.get("appeared") is not True and row.get("appeared") is not False:
            raise ValueError("appeared must be a boolean, including explicit DNP")
        if not row.get("source"):
            raise ValueError("roster provenance required")
        observed = parse_utc(row.get("roster_observed_at_utc"))
        tip = parse_utc(row.get("tipoff_at_utc"))
        if observed >= tip:
            raise ValueError("roster observation not before tipoff")
        games.add(row["game_id"])
        outcomes[row["appeared"]] += 1
    if outcomes[False] == 0:
        raise ValueError("zero DNP rows: appearance-conditioned sample not acceptable")
    return {"games": len(games), "candidates": len(rows),
            "appeared": outcomes[True], "did_not_appear": outcomes[False],
            "appearance_rate": outcomes[True] / len(rows)}

def join_injury_status(candidates, snapshots):
    """Latest injury observation strictly before each candidate's tipoff.

    Returns a copy of candidates; no missing status is interpreted as healthy.
    """
    index = {}
    for item in snapshots:
        pid = item.get("player_id")
        stamp = parse_utc(item.get("captured_at_utc"))
        if pid is None:
            raise ValueError("injury player_id required")
        index.setdefault(pid, []).append((stamp, item.get("status") or "unknown"))
    for events in index.values():
        events.sort(key=lambda x: x[0])
    out = []
    for row in candidates:
        tip = parse_utc(row["tipoff_at_utc"])
        history = [x for x in index.get(row["player_id"], []) if x[0] < tip]
        enriched = dict(row)
        enriched["injury_status_pretip"] = history[-1][1] if history else None
        enriched["injury_observed_at_utc"] = history[-1][0].isoformat() if history else None
        out.append(enriched)
    return out
