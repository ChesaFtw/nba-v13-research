"""Research-only delayed NBA game-date residual quantile; not a betting trigger.

A caller must independently verify the timestamp of completed-game results.
Same-date labels are always excluded, even when already present in memory.
"""
from datetime import date, timedelta
from math import ceil, isfinite


def delayed_radius(history, game_date, *, days=14, lag=1, coverage=.9,
                   minimum=20, fallback=None):
    """Return (absolute-error radius, sample size) using only eligible past days.

    history: (ISO NBA game date, nonnegative absolute residual) pairs.
    lag=1 admits yesterday, lag=2 admits no later than two days ago.
    fallback is an independently frozen radius, never learned from test labels.
    """
    if days<1 or lag<1 or minimum<20 or not 0<coverage<1:
        raise ValueError('invalid calibration parameters')
    target=date.fromisoformat(game_date)
    upper=target-timedelta(days=lag-1)
    lower=upper-timedelta(days=days)
    values=[]
    for raw_day,raw_error in history:
        d=date.fromisoformat(raw_day)
        error=float(raw_error)
        if d>target or not isfinite(error) or error<0:
            raise ValueError('invalid or future-dated residual')
        if lower<=d<upper:
            values.append(error)
    if len(values)<minimum:
        if fallback is None:raise ValueError('insufficient past labels')
        if not isfinite(float(fallback)) or fallback<0:
            raise ValueError('invalid fallback')
        return float(fallback),len(values)
    values.sort()
    rank=min(len(values),ceil((len(values)+1)*coverage))-1
    radius=values[rank]
    if fallback is not None:
        if not isfinite(float(fallback)) or fallback<0:
            raise ValueError('invalid fallback')
        radius=max(radius,float(fallback))
    return radius,len(values)
