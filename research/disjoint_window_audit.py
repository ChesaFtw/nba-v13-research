"""Fail-closed, disjoint chronological audit of scored pre-tip predictions.

Input rows contain actual, baseline, challenger, game_id, record_id,
target_date (YYYY-MM-DD), tipoff_at, predicted_at, trained_through.
This checks stated timestamps, not external authenticity. Not a betting model.
"""
from datetime import date, datetime, timezone
from research.nba_calendar_date import game_date_eastern
from research.paired_market_audit import paired_mae

DEFAULT_WINDOWS = (
    ("2026-01-15", "2026-02-15"),
    ("2026-02-15", "2026-03-12"),
    ("2026-03-12", "2026-04-13"),
)

def _utc(value):
    if not isinstance(value, str):
        raise ValueError("timestamp required")
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is None or d.utcoffset() is None:
        raise ValueError("timezone-aware timestamp required")
    return d.astimezone(timezone.utc)

def audit_disjoint_windows(rows, windows=DEFAULT_WINDOWS, *, iterations=2000):
    """All rows must be pre-tip predictions with prior-day training cutoff."""
    windows = tuple(windows)
    if not windows:
        raise ValueError("windows required")
    previous_end = None
    for start, end in windows:
        if date.fromisoformat(start) >= date.fromisoformat(end):
            raise ValueError("invalid window")
        if previous_end is not None and start < previous_end:
            raise ValueError("overlapping windows")
        previous_end = end
    bins = {start: [] for start, _ in windows}
    seen = set()
    rows = list(rows)
    for row in rows:
        rid = row.get("record_id")
        if rid is None or rid in seen:
            raise ValueError("missing or duplicate record_id")
        seen.add(rid)
        target = date.fromisoformat(row["target_date"])
        if date.fromisoformat(row["trained_through"]) >= target:
            raise ValueError("training date is not prior to target")
        tip = _utc(row["tipoff_at"])
        predicted = _utc(row["predicted_at"])
        if predicted >= tip:
            raise ValueError("prediction not before tipoff")
        if row['target_date'] != game_date_eastern(row['tipoff_at']):
            raise ValueError("target_date differs from NBA Eastern game date")
        for start, end in windows:
            if start <= row["target_date"] < end:
                bins[start].append(row)
                break
    return {
        "windows": {
            start: {"end_exclusive": end,
                    **paired_mae(bins[start], iterations=iterations)}
            for start, end in windows
        },
        "input_rows": len(rows),
        "evaluated_rows": sum(len(x) for x in bins.values()),
        "note": "Timestamp schema checks do not verify independent data provenance, training leakage, or market profitability.",
    }
