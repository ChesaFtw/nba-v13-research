import pytest
from research.paired_market_audit import paired_mae


def data(delta=1):
    return [{'game_id':i,'actual':10,'baseline':5,'challenger':5+delta}
            for i in range(10) for _ in range(2)]


def test_positive_improvement():
    r=paired_mae(data(),iterations=300)
    assert r['rows']==20 and r['games']==10
    assert r['improvement']==1 and r['ci95_game_cluster']==[1,1]


def test_worse_challenger():
    r=paired_mae(data(-1),iterations=300)
    assert r['improvement']==-1 and r['ci95_game_cluster']==[-1,-1]


def test_missing_game():
    x=data();x[0]['game_id']=None
    with pytest.raises(ValueError,match='game_id'):paired_mae(x)


def test_nonfinite():
    x=data();x[0]['actual']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):paired_mae(x)


def test_insufficient_games():
    with pytest.raises(ValueError,match='five'):paired_mae(data()[:4])


def test_reproducible():
    assert paired_mae(data(),seed=3,iterations=300)==paired_mae(data(),seed=3,iterations=300)
