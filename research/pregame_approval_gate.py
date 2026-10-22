"""Strict pregame model approval gate.

A betting recommendation must be supported by a market-specific, time-ordered,
out-of-sample calibrated probability. Statistical scenario means do NOT qualify.
No automatic betting or guarantee of positive expectation.
"""
from datetime import datetime, timezone
from math import isfinite


def _utc(t):
    d = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
    if d.tzinfo is None or d.utcoffset() is None:
        raise ValueError("Offset-aware timestamp required")
    return d.astimezone(timezone.utc)


def american_break_even(odds):
    o = int(odds)
    if o == 0 or -100 < o < 100:
        raise ValueError("Invalid American odds")
    return -o / (100 - o) if o < 0 else 100 / (100 + o)


def review(row, now=None):
    """Fail closed on missing material inputs, leakage, stale lineup, weak edge.

    row keys documented in research/FULL_GAME_MODEL_READINESS.md.
    p_lower is a conservative uncertainty bound on calibrated win probability,
    NOT an arbitrary confidence label or historic raw hit rate.
    """
    reasons = []
    mandatory = [
        "game_id", "market", "selection", "odds", "quote_at",
        "tipoff_at", "injuries_at", "projected_lineup_at",
        "features_asof", "train_cutoff", "training_games",
        "holdout_games", "calibration_mae", "p_lower",
        "probability_calibrated", "lineup_confidence",
        "new_rotation_share", "model_version",
    ]
    for k in mandatory:
        if row.get(k) is None or row.get(k) == "":
            reasons.append(f"missing_{k}")
    if reasons:
        return {"approved": False, "reasons": reasons}
    try:
        now = _utc(now or datetime.now(timezone.utc).isoformat())
        quote = _utc(row["quote_at"])
        tip = _utc(row["tipoff_at"])
        inj = _utc(row["injuries_at"])
        lineup = _utc(row["projected_lineup_at"])
        features = _utc(row["features_asof"])
        trained = _utc(row["train_cutoff"])
        if not (features < quote < tip and trained <= features):
            reasons.append("temporal_leakage_or_stale_features")
        if not (inj < tip and lineup < tip):
            reasons.append("post_tip_info")
        if (now - quote).total_seconds() > 45 * 60:
            reasons.append("stale_quote")
        if (now - inj).total_seconds() > 6 * 3600:
            reasons.append("stale_injury_report")
        if (now - lineup).total_seconds() > 6 * 3600:
            reasons.append("stale_expected_lineup")
        if int(row["training_games"]) < 1000:
            reasons.append("too_few_training_games")
        if int(row["holdout_games"]) < 300:
            reasons.append("too_few_holdout_games")
        if not bool(row["probability_calibrated"]):
            reasons.append("uncalibrated_probability")
        if float(row["calibration_mae"]) > .055:
            reasons.append("failed_calibration")
        if float(row["lineup_confidence"]) < .85:
            reasons.append("uncertain_minutes_or_lineup")
        if float(row["new_rotation_share"]) > .30:
            reasons.append("extreme_roster_churn_no_certified_adjustment")
        lower = float(row["p_lower"])
        if not isfinite(lower) or not 0 <= lower <= 1:
            reasons.append("invalid_probability_lower_bound")
        elif lower < american_break_even(row["odds"]) + .03:
            reasons.append("insufficient_lower_bound_edge")
        if str(row["market"]).lower() in ("spread", "moneyline"):
            # Never bet against the Knicks as explicitly requested.
            if str(row.get("opponent", "")).upper() in ("NYK", "NEW YORK KNICKS"):
                reasons.append("knicks_opponent_betting_rule")
    except (ValueError, TypeError, OverflowError):
        reasons.append("invalid_quantitative_inputs")
    return {"approved": not reasons, "reasons": reasons}
