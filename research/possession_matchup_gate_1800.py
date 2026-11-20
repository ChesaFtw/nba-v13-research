"""Offline V13 research helper: opponent pace-allowance blend; not production."""
from math import isfinite


def matchup_possessions(own_for, opponent_allowed, window=10):
    """Blend last-window own pace and opponent allowed pace, equally weighted.

    All inputs must be strictly prior games; caller enforces chronology.
    The equivalent FGA/FTA/OREB/TOV decomposition is algebraically identical.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError('invalid window')
    if len(own_for) < window or len(opponent_allowed) < window:
        raise ValueError('insufficient historical games')
    values = list(own_for[-window:]) + list(opponent_allowed[-window:])
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x) or x <= 0 for x in values):
        raise ValueError('invalid possessions')
    return (sum(own_for[-window:]) + sum(opponent_allowed[-window:])) / (2 * window)
