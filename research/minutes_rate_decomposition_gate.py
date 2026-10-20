"""Research-only lagged 3PA/minute and postgame-minutes decomposition guard.

`pregame_attempts` NEVER uses realized target-game minutes or box score.
`hindsight_components` explicitly uses realized minutes/attempts for diagnostic only.
"""
from datetime import date
from math import isfinite
from scipy.stats import nbinom

R = 16.0
FLOOR = 0.05


def _number(row, key, low, high, positive=False):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f'invalid {key}')
    if not (low <= value <= high) or (positive and value <= 0):
        raise ValueError(f'out-of-range {key}')
    return float(value)


def lagged_attempt_features(row):
    try:
        date.fromisoformat(row['date'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('invalid target date') from exc
    prior = _number(row, 'prior_minutes', 0, 65, True)
    short = _number(row, 'short_minutes', 0, 65, True)
    mu = _number(row, 'challenger_att', 0, 100)
    projected = .5 * (prior + short)
    if not 0 < projected <= 65:
        raise ValueError('invalid projected minutes')
    frozen_mu = max(FLOOR, mu)
    return frozen_mu, projected, frozen_mu / projected


def pregame_attempts(row):
    """Strictly lagged prediction; target-game fields ignored."""
    return lagged_attempt_features(row)[0]


def pregame_tail_six(row):
    mu = max(FLOOR, pregame_attempts(row))
    return float(nbinom.sf(5, R, R / (R + mu)))


def hindsight_components(row):
    """NOT PREGAME. Uses target-game minutes and 3PA, only for diagnostics."""
    mu, projected, rate = lagged_attempt_features(row)
    actual_min = _number(row, 'min', 0, 65, True)
    att = row.get('att')
    if type(att) is not int or not 0 <= att <= 60:
        raise ValueError('invalid realized 3PA')
    minutes_component = rate * (actual_min - projected)
    rate_component = att - rate * actual_min
    hindsight_mu = max(FLOOR, rate * actual_min)
    hindsight_tail = float(nbinom.sf(5, R, R / (R + hindsight_mu)))
    return {
        'minutes_error': actual_min - projected,
        'minutes_component': minutes_component,
        'rate_component': rate_component,
        'attempt_error': att - mu,
        'six_event': float(att >= 6),
        'tail_pregame': pregame_tail_six(row),
        'tail_hindsight': hindsight_tail,
    }
