import pytest
from research.role_volatility_gate import role_label, low_stable_widen, prior_day_radius

def test_role_cutoffs():
    assert role_label(14.99,7.99)==("low","stable")
    assert role_label(15,8)==("mid","volatile")
    assert role_label(30,0)==("high","stable")

def test_low_stable_only_widens():
    assert low_stable_widen(8,12,12,4)==12
    assert low_stable_widen(8,5,12,4)==8
    assert low_stable_widen(8,12,35,4)==8
    assert low_stable_widen(8,12,12,9)==8

def test_exclude_same_day_labels():
    h=[("2026-03-14",1.)]*20+[("2026-03-15",1000.)]*100
    assert prior_day_radius(h,"2026-03-15",minimum=20)==(1.,20)

def test_reject_future_labels():
    with pytest.raises(ValueError):
        prior_day_radius([("2026-03-16",1.)],"2026-03-15",minimum=20)

def test_exclude_old_labels_and_floor():
    h=[("2026-02-01",100.)]*100+[("2026-03-14",1.)]*20
    assert prior_day_radius(h,"2026-03-15",minimum=20,floor=4)==(4.,20)

def test_minimum_fallback():
    assert prior_day_radius([("2026-03-14",1.)]*19,"2026-03-15",minimum=20,floor=9)==(9.,19)

def test_bad_inputs():
    with pytest.raises(ValueError):role_label(float("nan"),1)
    with pytest.raises(ValueError):low_stable_widen(-1,2,12,4)
    with pytest.raises(ValueError):prior_day_radius([],"2026-03-15",days=0)
