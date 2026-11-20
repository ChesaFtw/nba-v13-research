"""Research-only, appearance-conditional three-made-threes diagnostic.

Pregame predictions depend exclusively on lagged features; realized attempts
are accepted ONLY by the explicitly labeled hindsight function.
"""
from math import isfinite
from datetime import date
from scipy.stats import nbinom, binom

R = 16.0
FLOOR = 0.05


def checked_lagged_features(row):
    """Return only lagged prediction features, never target-game outcomes."""
    try:
        date.fromisoformat(row['date'])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('invalid prediction date') from exc
    out = []
    for name in ('challenger_att', 'shooting_pct', 'prior_minutes'):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError('invalid lagged feature')
        out.append(float(value))
    mu, shooting, minutes = out
    if not (0 <= mu <= 100 and 0 <= shooting <= 1 and 0 < minutes <= 65):
        raise ValueError('lagged feature out of range')
    return max(FLOOR, mu), shooting, minutes


def pregame_three_plus(row):
    """Frozen NB16 prediction; target-game attempts/makes/minutes are ignored."""
    mu, shooting, _ = checked_lagged_features(row)
    expected_makes = mu * shooting
    return float(nbinom.sf(2, R, R / (R + expected_makes)))


def hindsight_attempt_three_plus(row):
    """NOT PREGAME: uses actual target-game 3PA to isolate calibration components."""
    _, shooting, _ = checked_lagged_features(row)
    att = row.get('att')
    if type(att) is not int or not 0 <= att <= 60:
        raise ValueError('invalid realized attempts')
    return float(binom.sf(2, att, shooting))


def checked_outcome(row):
    att = row.get('att')
    made = row.get('made')
    if type(att) is not int or type(made) is not int or not 0 <= made <= att <= 60:
        raise ValueError('invalid realized three-point counts')
    return att, made
