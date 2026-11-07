"""Offline, appearance-conditional three-point probability diagnostics only.

No model activation, odds handling, staking, or wager execution.
"""
from datetime import date
from math import expm1, isfinite


def prior_event_probabilities(n, hit_count, made_mean):
    """Return Poisson P(3PM>=1) and Beta(1,1) smoothed empirical probability."""
    if (type(n) is not int or not 5 <= n <= 10
            or type(hit_count) is not int or not 0 <= hit_count <= n
            or isinstance(made_mean, bool)):
        raise ValueError("invalid prior appearance sample")
    try:
        mean = float(made_mean)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid historical made-three rate") from exc
    if not isfinite(mean) or mean < 0:
        raise ValueError("invalid historical made-three rate")
    return -expm1(-mean), (hit_count + 1) / (n + 2)


def checked_prior_dates(target_date, observation_dates):
    """Reject same-day and future history."""
    try:
        target = date.fromisoformat(target_date) if isinstance(target_date, str) else target_date
        if type(target) is not date:
            raise ValueError("target must be an ISO date")
        observations = list(observation_dates)
        for raw in observations:
            observed = date.fromisoformat(raw) if isinstance(raw, str) else raw
            if type(observed) is not date or observed >= target:
                raise ValueError("history includes same-day or future date")
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid or non-prior observation date") from exc
    return len(observations)


def brier_score(actual, predicted):
    """Average squared error of binary probability forecasts."""
    try:
        y, p = list(actual), list(predicted)
    except TypeError as exc:
        raise ValueError("expected nonempty sequences") from exc
    if not y or len(y) != len(p):
        raise ValueError("missing or misaligned predictions")
    errors = []
    for outcome, probability in zip(y, p):
        if isinstance(outcome, bool) or isinstance(probability, bool):
            raise ValueError("invalid binary outcome or probability")
        try:
            outcome, probability = float(outcome), float(probability)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid binary outcome or probability") from exc
        if (not isfinite(outcome) or outcome not in (0.0, 1.0)
                or not isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("invalid binary outcome or probability")
        errors.append((outcome - probability) ** 2)
    return sum(errors) / len(errors)
