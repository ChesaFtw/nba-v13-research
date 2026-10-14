import pytest
from scripts.grade_historical_selections import grade

def row(side="over",actual=25,line=20):
    return dict(game_id=1,market="points",side=side,actual=actual,line=line,
        probability=.6,model_version="v13-test",line_source="fixture",
        prediction_time="2026-03-01T17:00:00Z",
        line_observed_at="2026-03-01T16:00:00Z",
        game_time="2026-03-01T23:00:00Z")

def test_win_and_compounding():
    result=grade([row()],10000,.01)
    assert result["record"]["win"]==1
    assert result["end_bankroll"]==10090.91

def test_loss_and_push():
    assert grade([row("under")])["record"]["loss"]==1
    assert grade([row(actual=20)])["record"]["push"]==1

def test_posttip_rejected():
    r=row();r["line_observed_at"]="2026-03-02T00:00:00Z"
    with pytest.raises(ValueError):grade([r])

def test_missing_threshold_rejected():
    r=row();del r["line"]
    with pytest.raises(ValueError):grade([r])

def test_duplicate_rejected():
    with pytest.raises(ValueError):grade([row(),row()])
