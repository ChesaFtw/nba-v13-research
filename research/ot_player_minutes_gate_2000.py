"""Research-only strict chronology and workload gate. No production integration."""
from datetime import date
from math import isfinite


def classify_previous_ot_workload(*, target_date, previous_team_date, previous_team_game_id,
                                  previous_player_date, previous_player_game_id,
                                  previous_player_team, current_team, previous_minutes,
                                  previous_team_overtime):
    """Return (heavy_short_rest, ot_heavy_short_rest, days_off), or None."""
    if not isinstance(previous_team_overtime, (bool, int)) or previous_team_overtime not in (False, True, 0, 1):
        raise ValueError("invalid overtime flag")
    if isinstance(previous_minutes, bool) or not isinstance(previous_minutes, (int, float)) or not isfinite(previous_minutes) or not (0 < previous_minutes <= 65):
        raise ValueError("invalid previous minutes")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in
           (previous_team_game_id, previous_player_game_id, previous_player_team, current_team)):
        raise ValueError("invalid game or team identifier")
    t = date.fromisoformat(target_date)
    a = date.fromisoformat(previous_team_date)
    b = date.fromisoformat(previous_player_date)
    if not a < t or not b < t:
        raise ValueError("previous dates must precede target")
    if previous_player_team != current_team or previous_player_game_id != previous_team_game_id or a != b:
        return None
    days_off = (t-a).days-1
    if days_off > 10:
        return None
    heavy_short = previous_minutes >= 36 and days_off <= 1
    return heavy_short, heavy_short and bool(previous_team_overtime), days_off


def safe_minutes_prediction(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError("invalid minutes forecast")
    return max(0.0, min(65.0, float(value)))
