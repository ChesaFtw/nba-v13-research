import pytest
from research.three_event_calibration_gate import prior_event_probabilities, checked_prior_dates, brier_score

def test_zero_prior(): assert prior_event_probabilities(5,0,0)==(0.0,1/7)
def test_full_prior():
    a,b=prior_event_probabilities(10,10,2)
    assert 0<a<1 and b==11/12
@pytest.mark.parametrize("args",[(0,0,0),(5,6,0),(5,-1,0),(5,2,float("nan")),(11,0,0),(True,0,0)])
def test_reject_invalid_prior(args):
    with pytest.raises(ValueError): prior_event_probabilities(*args)
def test_strict_prior_dates():
    assert checked_prior_dates("2026-02-01",["2026-01-31"])==1
    with pytest.raises(ValueError): checked_prior_dates("2026-02-01",["2026-02-01"])
    with pytest.raises(ValueError): checked_prior_dates("2026-02-01",["2026-02-02"])
def test_brier_score():
    assert brier_score([0,1],[0,1])==0
    assert brier_score([0,1],[1,0])==1
def test_bad_brier():
    for a,p in [([],[]),([1],[1.1]),([2],[.3]),([0],[float("nan")])]:
        with pytest.raises(ValueError): brier_score(a,p)
