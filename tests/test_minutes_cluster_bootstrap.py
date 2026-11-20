import pytest
from research.minutes_cluster_bootstrap import audit_minutes

def rows(better=True):
    return [dict(game_id=f"G{i}",actual=30,baseline=25 if better else 29,
                 challenger=29 if better else 25,
                 predicted_at_utc="2026-10-20T16:00:00Z",
                 tipoff_at_utc="2026-10-20T23:00:00Z") for i in range(10)]

def test_improves():
    r=audit_minutes(rows(),repetitions=300)
    assert r["mae_improvement"]==4 and r["ci95"]==(4,4) and r["lower_bound_positive"]
def test_worse():
    r=audit_minutes(rows(False),repetitions=300)
    assert r["mae_improvement"]==-4 and not r["lower_bound_positive"]
def test_few_games():
    with pytest.raises(ValueError,match="five independent"):audit_minutes(rows()[:4])
def test_missing_game_id():
    r=rows();del r[0]["game_id"]
    with pytest.raises(ValueError,match="game_id"):audit_minutes(r)
def test_post_tipoff():
    r=rows();r[0]["predicted_at_utc"]=r[0]["tipoff_at_utc"]
    with pytest.raises(ValueError,match="precede"):audit_minutes(r)
def test_nonfinite():
    r=rows();r[0]["baseline"]=float("nan")
    with pytest.raises(ValueError,match="finite"):audit_minutes(r)
def test_reproducible():
    assert audit_minutes(rows(),repetitions=300,seed=4)==audit_minutes(rows(),repetitions=300,seed=4)
