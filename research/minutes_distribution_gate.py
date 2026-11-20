"""Offline research-only 3PA mean propagation; no production or betting hooks."""
from datetime import date
from math import isfinite

FLOOR = 0.05

def validate_prediction_row(row, oracle=False):
    required = ('date','baseline_att','challenger_att','prior_minutes','att','made','shooting_pct')
    if any(key not in row for key in required):
        raise ValueError('missing field')
    try:
        date.fromisoformat(row['date'])
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid target date') from exc
    for name in ('baseline_att','challenger_att','prior_minutes','shooting_pct'):
        v = row[name]
        if isinstance(v, bool) or not isinstance(v, (int,float)) or not isfinite(v):
            raise ValueError('invalid lagged feature')
    if not (0 <= row['baseline_att'] <= 100 and 0 <= row['challenger_att'] <= 100
            and 0 < row['prior_minutes'] <= 65 and 0 <= row['shooting_pct'] <= 1):
        raise ValueError('invalid lagged feature range')
    if type(row['att']) is not int or type(row['made']) is not int or not 0 <= row['made'] <= row['att'] <= 60:
        raise ValueError('invalid realized count')
    if oracle:
        if ('min' not in row or isinstance(row['min'], bool)
                or not isinstance(row['min'], (int,float))
                or not isfinite(row['min']) or not 0 < row['min'] <= 65):
            raise ValueError('invalid realized minutes')
    return True

def means(row, include_hindsight=False):
    validate_prediction_row(row, oracle=include_hindsight)
    out = {'last10': max(FLOOR, float(row['baseline_att'])),
           'minutes_blend': max(FLOOR, float(row['challenger_att']))}
    if include_hindsight:
        out['hindsight_actual_minutes'] = max(
            FLOOR, float(row['baseline_att']) / float(row['prior_minutes']) * float(row['min']))
    return out
