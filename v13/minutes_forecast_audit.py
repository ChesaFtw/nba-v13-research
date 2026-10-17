"""Research-only tools for evaluating minutes forecasts without look-ahead bias.

Use this module to compare candidate minutes models against a frozen baseline.
It does not generate betting recommendations or alter production models.
"""
from math import isfinite


def evaluate_minutes_forecasts(rows):
    """Evaluate rows with actual, baseline, challenger and strictly pregame timestamp.

    Each row must contain numerical 'actual', 'baseline', 'challenger',
    'predicted_at_utc' and 'tipoff_at_utc'. Returns MAE and coverage.
    Raises on invalid or post-tipoff predictions rather than silently scoring them.
    """
    from datetime import datetime

    def parse_utc(value):
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("Timestamps must include a timezone")
        return dt

    rows = list(rows)
    if not rows:
        raise ValueError("At least one prediction is required")
    baseline_errors, challenger_errors = [], []
    for row in rows:
        if parse_utc(row["predicted_at_utc"]) >= parse_utc(row["tipoff_at_utc"]):
            raise ValueError("Prediction must be recorded before tipoff")
        vals = [float(row[key]) for key in ("actual", "baseline", "challenger")]
        if not all(isfinite(v) for v in vals):
            raise ValueError("Minutes values must be finite")
        actual, baseline, challenger = vals
        if not 0 <= actual <= 60:
            raise ValueError("Actual minutes outside plausible NBA game range")
        baseline_errors.append(abs(actual - baseline))
        challenger_errors.append(abs(actual - challenger))
    n = len(rows)
    old = sum(baseline_errors) / n
    new = sum(challenger_errors) / n
    return {"n": n, "baseline_mae": old, "challenger_mae": new,
            "mae_improvement": old - new, "challenger_better": new < old}
