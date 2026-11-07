import pytest
from research.pregame_tail_risk_gate import tail_risk_features

def row():
    return {'x':[10,10,10,14,4,0,3,1,0,1,0,15]}

def test_scores_known_values():
    scores=tail_risk_features(row())
    assert scores['volatility']==4
    assert scores['last_game_jump']==4
    assert scores['appearance_gap']==3
    assert scores['composite']==pytest.approx(6.7)

def test_only_uses_pregame_features():
    r=row(); r['minutes_actual']=99; r['date']='2099-01-01'
    assert tail_risk_features(r)==tail_risk_features(row())

def test_gap_capped():
    r=row();r['x'][6]=20
    assert tail_risk_features(r)['composite']==pytest.approx(9.5)

def test_team_change_adds_five():
    r=row();r['x'][10]=1
    assert tail_risk_features(r)['composite']==pytest.approx(11.7)

def test_rejects_nonfinite():
    r=row();r['x'][4]=float('nan')
    with pytest.raises(ValueError):tail_risk_features(r)

def test_rejects_negative():
    r=row();r['x'][6]=-1
    with pytest.raises(ValueError):tail_risk_features(r)

def test_rejects_malformed():
    with pytest.raises(ValueError):tail_risk_features({'x':[1]})
