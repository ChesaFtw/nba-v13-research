import pytest
from scripts.grade_spreads_totals import evaluate

def sample(market="spread",prediction=8,line=-4.5,home=110,away=105):
    return dict(game_id=1,game_time="2026-03-02T01:00:00Z",prediction_time="2026-03-01T22:00:00Z",
        line_observed_at="2026-03-01T21:00:00Z",line_source="fixture",model_version="test",
        market=market,prediction=prediction,line=line,home_score=home,away_score=away)

def test_spread_home_cover():
    r=evaluate([sample()])
    assert r["markets"]["spread"]["wins"]==1
    assert r["ending_bankroll"]==10090.91

def test_spread_away_cover():
    r=evaluate([sample(prediction=1)])
    assert r["markets"]["spread"]["losses"]==1

def test_total_under():
    r=evaluate([sample(market="total",prediction=210,line=220)])
    assert r["markets"]["total"]["wins"]==1

def test_push():
    r=evaluate([sample(line=-5)])
    assert r["markets"]["spread"]["pushes"]==1

def test_missing_market_line():
    row=sample();del row["line"]
    with pytest.raises(ValueError):evaluate([row])

def test_posttip_line_rejected():
    row=sample();row["line_observed_at"]="2026-03-02T02:00:00Z"
    with pytest.raises(ValueError):evaluate([row])

def test_no_edge_pass():
    r=evaluate([sample(prediction=4.5)])
    assert r["markets"]["spread"]["bets"]==0
