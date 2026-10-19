"""Research-only regression tests: prior overtime / heavy workload chronology."""
import pytest
from research.ot_player_minutes_gate_2000 import classify_previous_ot_workload as f, safe_minutes_prediction

BASE = dict(target_date='2025-02-04',previous_team_date='2025-02-02',
            previous_team_game_id=15,previous_player_date='2025-02-02',
            previous_player_game_id=15,previous_player_team=2,current_team=2,
            previous_minutes=36,previous_team_overtime=1)

def test_one_day_off_exposed():
    assert f(**BASE)==(True,True,1)

def test_non_ot_control():
    assert f(**{**BASE,'previous_team_overtime':0})==(True,False,1)

def test_two_days_off_not_exposed():
    assert f(**{**BASE,'target_date':'2025-02-05'})==(False,False,2)

def test_just_below_heavy_cutoff():
    assert f(**{**BASE,'previous_minutes':35.99})==(False,False,1)

def test_missing_previous_game_dnp_excluded():
    assert f(**{**BASE,'previous_player_game_id':14}) is None

def test_team_change_excluded():
    assert f(**{**BASE,'previous_player_team':3}) is None

def test_same_day_player_leakage_rejected():
    with pytest.raises(ValueError):f(**{**BASE,'previous_player_date':'2025-02-04'})

def test_future_team_game_rejected():
    with pytest.raises(ValueError):f(**{**BASE,'previous_team_date':'2025-02-05'})

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,66,True])
def test_invalid_minutes_rejected(value):
    with pytest.raises(ValueError):f(**{**BASE,'previous_minutes':value})

@pytest.mark.parametrize('value',[-1,70,35.5])
def test_forecast_clipping(value):
    assert 0<=safe_minutes_prediction(value)<=65

def test_invalid_overtime_flag_rejected():
    with pytest.raises(ValueError):f(**{**BASE,'previous_team_overtime':2})
