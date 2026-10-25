"""Offline research-only volatility thresholds by prior playing-time role.

No live selection, odds, staking or model activation. Thresholds are frozen
using earlier-dated observations; appearance probabilities are not inferred.
"""
from datetime import date
from math import ceil, isfinite

ROLES = ('under15', '15to30', '30plus')


def role_for_minutes(value):
    value = float(value)
    if not isfinite(value) or value < 0 or value > 53:
        raise ValueError('invalid historical minutes')
    return 'under15' if value < 15 else ('15to30' if value < 30 else '30plus')


def frozen_role_thresholds(calibration, evaluation_date, *, fraction=0.8, min_count=50):
    """Earlier-dated quantiles; missing roles return None, not a low-risk flag."""
    target = date.fromisoformat(evaluation_date)
    if not 0 < fraction < 1 or min_count < 20:
        raise ValueError('invalid threshold configuration')
    groups = {k: [] for k in ROLES}
    for row in calibration:
        d = date.fromisoformat(row['date'])
        if d >= target:
            raise ValueError('same-day/future calibration row')
        v = float(row['recent_minutes_std'])
        if not isfinite(v) or v < 0:
            raise ValueError('invalid volatility')
        groups[role_for_minutes(row['minutes_baseline'])].append(v)
    result = {}
    for role, vals in groups.items():
        if len(vals) < min_count:
            result[role] = None
        else:
            vals.sort()
            result[role] = vals[max(0, ceil(fraction * len(vals)) - 1)]
    return result


def high_risk(row, thresholds):
    role = role_for_minutes(row['minutes_baseline'])
    cutoff = thresholds[role]
    if cutoff is None:
        return None
    v = float(row['recent_minutes_std'])
    if not isfinite(v) or v < 0:
        raise ValueError('invalid volatility')
    return v >= cutoff
