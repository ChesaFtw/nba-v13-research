import pytest
from research.disjoint_window_audit import audit_disjoint_windows

def row(i, date="2026-01-20"):
    return {"record_id":i,"game_id":i,"target_date":date,
            "tipoff_at":date+"T23:00:00Z",
            "predicted_at":date+"T20:00:00Z",
            "trained_through":"2026-01-14",
            "actual":10,"baseline":6,"challenger":9}

def test_two_nonoverlapping_windows():
    rows=[row(i) for i in range(6)]+[row(i+6,"2026-02-20") for i in range(6)]
    result=audit_disjoint_windows(rows,[("2026-01-15","2026-02-15"),("2026-02-15","2026-03-12")],iterations=200)
    assert result["evaluated_rows"]==12
    assert all(v["improvement"]==3 for v in result["windows"].values())

def test_overlapping_windows_rejected():
    with pytest.raises(ValueError,match="overlapping"):
        audit_disjoint_windows([], [("2026-01-15","2026-02-20"),("2026-02-15","2026-03-12")])

def test_post_tip_prediction_rejected():
    r=row(1);r["predicted_at"]="2026-01-20T23:01:00Z"
    with pytest.raises(ValueError,match="before tipoff"):
        audit_disjoint_windows([r])

def test_same_day_training_rejected():
    r=row(1);r["trained_through"]="2026-01-20"
    with pytest.raises(ValueError,match="training date"):
        audit_disjoint_windows([r])

def test_duplicate_record_rejected():
    with pytest.raises(ValueError,match="duplicate"):
        audit_disjoint_windows([row(1),row(1)])

def test_fewer_than_five_games_rejected():
    with pytest.raises(ValueError,match="five"):
        audit_disjoint_windows([row(i) for i in range(4)],iterations=200)
